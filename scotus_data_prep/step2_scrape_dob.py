import argparse
import datetime
import json
import os
import pickle
import re
import ssl
import string
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np
import requests
import wikipedia
import wptools
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from Levenshtein import distance as lev_dist
from local_info import (AVVO_API_ACCESS_TOKEN, AVVO_CSE_ID, GOOGLE_API_KEY,
                        WIKI_CSE_ID)
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    parser = argparse.ArgumentParser(description='Reads through speaker_ids.json and tries to webscrape the DOB of each lawyer')
    parser.add_argument('--base-outfolder', type=str, help='Location of the base outfolder')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite dob pickle (default: False)')
    parser.add_argument('--skip-attempted', action='store_true', default=False, help='Skip names which have been attempted before')
    args = parser.parse_args()
    return args

class LawyerDOBParser:

    def __init__(self, graduation_dob_offset=25, distance_threshold=4, minimum_age=18):
        self.graduation_dob_offset = graduation_dob_offset
        self.distance_threshold = distance_threshold
        self.scotus_clerks_populated = False
        self.minimum_age = minimum_age
        self.minimum_dob = datetime.datetime(2005, 10, 1) - relativedelta(years=minimum_age)

        self.google_api_key = GOOGLE_API_KEY
        self.avvo_cse_id = AVVO_CSE_ID
        self.wiki_cse_id = WIKI_CSE_ID

    def parse_name(self, name):
        '''
        Parse name looking at wikipedia, then SCOTUS clerks, then the JUSTIA website

        Input: name
        Output: datetime object for D.O.B
        '''
        if not self.scotus_clerks_populated:
            self.get_scotus_clerks()

        print('Searching for DOB of {}....'.format(name))
        # Search wikipedia for person
        wiki_dob, wiki_info = self.parse_wiki(name)
        if wiki_dob:
            if wiki_dob <= self.minimum_dob:
                return wiki_dob, wiki_info

        # Search through supreme court clerks for person
        scotus_dob, scotus_info = self.search_scotus_clerks(name)
        if scotus_dob:
            scotus_dob = datetime.datetime(scotus_dob, 7, 2)
            if scotus_dob <= self.minimum_dob:
                return scotus_dob, scotus_info

        # Search through JUSTIA website
        justia_dob, justia_info = self.parse_justia(name)
        if justia_dob:
            justia_dob = datetime.datetime(justia_dob, 7, 2)
            if justia_dob <= self.minimum_dob:
                return justia_dob, justia_info

        # Search through AVVO website
        avvo_dob, avvo_info = self.parse_avvo(name)
        if avvo_dob:
            time.sleep(1)
            avvo_dob = datetime.datetime(avvo_dob, 7, 2)
            if avvo_dob <= self.minimum_dob:
                return avvo_dob, avvo_info

        print("Couldn't find any age for {}".format(name))
        info_list = [wiki_info, scotus_info, justia_info, avvo_info]
        collated_info = {'info': {'type': None, 'error': 'no info found', 'collated_info': info_list}}
        return None, collated_info

    def parse_wiki(self, name):
        search = wikipedia.search(name)
        if search:
            if self.name_distance(name, search[0]) <= self.distance_threshold:
                name = search[0]

        wpage = wptools.page(name, silent=True)
        info = {'info': {'type': 'wiki', 'error': None, 'name': name}}
        try:
            page = wpage.get_parse()
        except:
            info['info']['error'] = 'page not found'
            return None, info
        try:
            if page.data:
                if 'infobox' in page.data:
                    if 'birth_date' in page.data['infobox']:
                        dob = page.data['infobox']['birth_date'].strip('{}').split('|')
                        dinfo = []
                        for d in dob:
                            try:
                                dinfo.append(int(d))
                            except:
                                continue
                        if dinfo:
                            if len(dinfo) > 3:
                                dinfo = dinfo[-3:]
                            if dinfo[0] > 1900:  # simple check if 4-digit year recognised
                                prelim_date = [1, 1, 1]
                                for i, d in enumerate(dinfo):
                                    prelim_date[i] = d
                                dob = datetime.datetime(*prelim_date)
                                info['info']['links'] = page.data['iwlinks']
                                return dob, info
            info['info']['error'] = 'page couldnt be parsed'
            return None, info
        except:
            info['info']['error'] = 'page couldnt be parsed'
            return None, info

    def parse_justia(self, name):
        searched_name, distance, justia_url = self.search_justia(name)
        info = {'info': {'type': 'justia', 'searched_name': searched_name, 'justia_url': justia_url, 'error': None}}
        if distance <= self.distance_threshold:
            grad_year = self.parse_justia_lawyer(justia_url)
            if grad_year:
                return grad_year - self.graduation_dob_offset, info
            else:
                info['info']['error'] = 'no year found'
                return None, info
        else:
            info['info']['error'] = 'distance threshold not met'
            return None, info

    def search_justia(self, name):
        """
        Input: Name to search, i.e. Anthony A. Yang (str,)
        Output: Matched name, Levenshtein distance to input, JUSTIA url
        """
        base_search_url = 'https://lawyers.justia.com/search?profile-id-field=&practice-id-field=&query={}&location='
        base_name = name.translate(str.maketrans('', '', string.punctuation)).lower()
        name_query = '+'.join(base_name.split())
        search_url = base_search_url.format(name_query)

        search_url = base_search_url.format(name_query)
        search_request = requests.get(search_url)
        soup = BeautifulSoup(search_request.content, 'lxml')
        lawyer_avatars = soup.findAll('a', attrs={'class': 'lawyer-avatar'})

        if lawyer_avatars:
            search_names = []
            search_urls = []

            for a in lawyer_avatars:
                search_names.append(a['title'])
                search_urls.append(a['href'])

            search_names = np.array(search_names)
            search_names_base = [n.translate(str.maketrans('', '', string.punctuation)).lower() for n in search_names]

            distances = np.array([self.name_distance(name, n) for n in search_names])
            search_urls = np.array(search_urls)

            dist_order = np.argsort(distances)
            distances = distances[dist_order]
            search_urls = search_urls[dist_order]
            search_names = search_names[dist_order]

            return search_names[0], distances[0], search_urls[0]
        else:
            return 'None', 100000, 'None'

    @staticmethod
    def parse_justia_lawyer(lawyer_url):
        """
        Input: Justia lawyer page url
        Output: Graduation year
        """
        r = requests.get(lawyer_url)
        soup = BeautifulSoup(r.content, 'lxml')

        jurisdictions = soup.find('div', attrs={'id': 'jurisdictions-block'})
        if jurisdictions:
            jd_admitted_year = []
            for j in jurisdictions:
                try:
                    jd_admitted_year.append(int(j.find('time')['datetime']))
                except:
                    continue
            if jd_admitted_year:
                return min(jd_admitted_year)
            else:
                # look for professional associations if jurisdictions is emtpy
                prof_assoc = None
                education = None
                blocks = soup.findAll('div', attrs={'class': 'block'})
                for block in blocks:
                    subdivs = block.findAll('div')
                    for subdiv in subdivs:
                        if subdiv.text == 'Professional Associations':
                            prof_assoc = block
                            break
                        if subdiv.text == 'Education':
                            education = block
                            break

                if prof_assoc:
                    prof_assoc_year = []
                    professional_associations = prof_assoc.findAll('time')
                    for p in professional_associations:
                        try:
                            prof_assoc_year.append(int(p['datetime']))
                        except:
                            continue
                    if prof_assoc_year:
                        return min(prof_assoc_year)

                if education:
                    education_years = []
                    education_history = education.findAll('dl')
                    for e in education_history:
                        degree_type = e.find('dd').text
                        if degree_type.strip().translate(str.maketrans('', '', string.punctuation)).lower() == 'jd':
                            try:
                                return int(e.find('time')['datetime'])
                            except:
                                continue

    def search_scotus_clerks(self, query_name):
        assert self.clerk_dob_dict, 'get_scotus_clerks must be called before this function'
        # query_name = query_name.translate(str.maketrans('', '', string.punctuation)).lower()
        distances = np.array([self.name_distance(query_name, k) for k in self.scotus_clerks])
        closest_match = np.argmin(distances)
        info = {'info': {'type': 'clerk', 'closest_match': closest_match, 'error': None}}
        if distances[closest_match] <= self.distance_threshold:
            return self.clerk_dob_dict[self.scotus_clerks[closest_match]], info
        else:
            info['info']['error'] = 'distance threshold not met'
            return None, info

    def get_scotus_clerks(self):
        """
        Populates self.clerk_dob_dict with dates of birth for SCOTUS clerks
        """
        base_url = 'https://en.wikipedia.org/wiki/List_of_law_clerks_of_the_Supreme_Court_of_the_United_States_({})'
        seats = ['Chief_Justice', 'Seat_1', 'Seat_2',
                 'Seat_3', 'Seat_4', 'Seat_6', 'Seat_8',
                 'Seat_9', 'Seat_10']
        urls = [base_url.format(s) for s in seats]

        self.all_cdicts = []
        self.clerk_dob_dict = OrderedDict({})

        for url in urls:
            mini_clerk_dict = self.parse_clerk_wiki(url)
            self.all_cdicts.append(mini_clerk_dict)

        for cdict in self.all_cdicts:
            self.clerk_dob_dict = {**self.clerk_dob_dict, **cdict}

        self.scotus_clerks = np.array(list(self.clerk_dob_dict.keys()))
        self.scotus_clerks_populated = True

    def parse_clerk_wiki(self, url):
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'lxml')
        tables = soup.findAll('table', attrs={'class': 'wikitable'})
        clerk_dict = {}
        for table in tables:
            for tr in table.findAll('tr'):
                row_entries = tr.findAll('td')
                if len(row_entries) != 5:
                    continue
                else:
                    name = row_entries[0].text
                    u = row_entries[3].text
                    year_candidates = re.findall(r'\d{4}', u)

                    if year_candidates:
                        year = int(year_candidates[0])
                    else:
                        continue

                    cleaned_name = re.sub(r'\([^)]*\)', '', name)
                    cleaned_name = re.sub(r'\[[^)]*\]', '', cleaned_name).strip()
                    clerk_dict[cleaned_name] = year - self.graduation_dob_offset

        return clerk_dict

    def parse_avvo(self, name):
        avvo_ids, response = self.get_avvo_ids_google(name)
        info = {'info': {'type': 'avvo', 'google_response': response, 'error': None}}
        if avvo_ids:
            for aid in avvo_ids:
                avvo_resp = self.get_from_avvo_api(aid)
                dob_estimate = self.parse_avvo_api_response(avvo_resp, name)
                if dob_estimate:
                    info['info']['avvo_id'] = aid
                    return dob_estimate, info
                else:
                    continue
            info['info']['error'] = 'no avvo ids yielded a dob estimate'
            info['info']['avvo_ids_attempted'] = avvo_ids
            return None, info
        else:
            info['info']['error'] = 'no links found.. check response'
            return None, info

    def parse_avvo_api_response(self, r, name):
        # r: response as dict
        if r['lawyers']:
            lawyer = r['lawyers'][0]
            licensed_since = lawyer['licensed_since']
            if licensed_since:
                lawyer_name = '{} {} {}'.format(lawyer['firstname'], lawyer['middlename'], lawyer['lastname'])
                if self.name_distance(name, lawyer_name) <= self.distance_threshold:
                    return licensed_since - self.graduation_dob_offset

    def search_google_avvo(self, query):
        r = requests.get('https://www.googleapis.com/customsearch/v1/siterestrict?key={}&cx={}&num=3&q="{}"'.format(
            self.google_api_key, self.avvo_cse_id, query))
        r = json.loads(r.content)
        if 'items' in r:
            links = [l['link'] for l in r['items']]
            return links, r
        else:
            return None, r

    def get_avvo_ids_google(self, name):
        links, response = self.search_google_avvo(name)
        if links:
            avvo_ids = [self.get_avvo_id_from_link(l) for l in links]
            avvo_ids = [i for i in avvo_ids if i]
            return avvo_ids, response
        else:
            return None, response

    @staticmethod
    def get_avvo_id_from_link(link):
        if link.startswith('https://www.avvo.com/attorneys/'):
            page_path = os.path.splitext(link.split('/')[-1])[0]
            avvo_id = page_path.split('-')[-1]
            if avvo_id.isnumeric():
                return avvo_id

    @staticmethod
    def get_from_avvo_api(avvo_id):
        headers = {'Authorization': 'Bearer {}'.format(AVVO_API_ACCESS_TOKEN)}
        url = 'https://api.avvo.com/api/4/lawyers/{}.json'.format(avvo_id)
        r = requests.get(url, headers=headers)
        return json.loads(r.content)

    @classmethod
    def name_distance(cls, string1, string2, wrong_initial_penalty=5):
        '''
        levenshtein distance accommodating for initials
        First and last initials must match
        TODO: allow for hyphenated names with last name partial match
        '''
        name1 = string1.lower().translate(str.maketrans('', '', string.punctuation)).lower()
        name2 = string2.lower().translate(str.maketrans('', '', string.punctuation)).lower()
        base_dist = lev_dist(name1, name2)

        if base_dist == 0:
            return 0

        if '-' in string1.split()[-1] or '-' in string2.split()[-1]:
            s1_perms = cls.hyphenation_perm(string1)
            s2_perms = cls.hyphenation_perm(string2)
            dists = []
            for s1 in s1_perms:
                for s2 in s2_perms:
                    dists.append(cls.name_distance(s1, s2))
            return min(dists)

        name1_split = name1.split()
        name2_split = name2.split()

        if name1_split[0] == name2_split[0] and name1_split[-1] == name2_split[-1]:
            if len(name1_split) == 2 or len(name2_split) == 2:
                return lev_dist(' '.join([name1_split[0], name1_split[-1]]),
                                ' '.join([name2_split[0], name2_split[-1]]))

            newname1 = ' '.join([n[0] if (1 <= i < len(name1_split) - 1) else n for i, n in enumerate(name1_split)])
            newname2 = ' '.join([n[0] if (1 <= i < len(name2_split) - 1) else n for i, n in enumerate(name2_split)])
            return lev_dist(newname1, newname2)
        else:
            return base_dist + wrong_initial_penalty

    @staticmethod
    def hyphenation_perm(name):
        splitup = name.split()
        lastname = splitup[-1]
        if '-' in lastname:
            lname_candidates = [' '.join(splitup[:-1] + [l]) for l in lastname.split('-')]
            return lname_candidates
        else:
            return [name]


if __name__ == "__main__":
    args = parse_args()
    base_outfolder = args.base_outfolder
    assert os.path.isdir(base_outfolder)

    pickle_path = os.path.join(base_outfolder, 'dob.p')
    info_pickle_path = os.path.join(base_outfolder, 'dob_info.p')

    speaker_id_path = os.path.join(base_outfolder, 'speaker_ids.json')
    assert os.path.isfile(speaker_id_path), "Can't find speaker_ids.json"

    speaker_ids = json.load(open(speaker_id_path, encoding='utf-8'),
                            object_pairs_hook=OrderedDict)

    parser = LawyerDOBParser()
    parser.get_scotus_clerks()

    if args.overwrite or not os.path.isfile(pickle_path):
        dobs = OrderedDict({})
        infos = OrderedDict({})
        speakers_to_scrape = sorted(speaker_ids.keys())
    else:
        infos = pickle.load(open(info_pickle_path, 'rb'))
        dobs = pickle.load(open(pickle_path, 'rb'))
        if args.skip_attempted:
            speakers_to_scrape = set(speaker_ids.keys()) - set(dobs.keys())
        else:
            speakers_to_scrape = set(speaker_ids.keys()) - set([s for s in dobs if dobs[s]])

        if speakers_to_scrape:
            speakers_to_scrape = sorted(list(speakers_to_scrape))

    for s in tqdm(speakers_to_scrape):
        query_name = speaker_ids[s]['name']
        parsed_dob, info = parser.parse_name(query_name)
        dobs[s] = parsed_dob
        infos[s] = info
        pickle.dump(dobs, open(pickle_path, 'wb'))
        pickle.dump(infos, open(info_pickle_path, 'wb'))

    print('Done!')

