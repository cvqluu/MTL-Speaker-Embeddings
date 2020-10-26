import argparse
import json
import os
import ssl
import sys
import urllib
from glob import glob

import numpy as np
import requests
import wget
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context
from collections import OrderedDict
from multiprocessing import Pool

import dateutil.parser as dparser


def parse_args():
    parser = argparse.ArgumentParser(description='Download mp3s and process the case jsons into something usable')
    parser.add_argument('--case-folder', type=str, help='Location of the case json files')
    parser.add_argument('--base-outfolder', type=str, help='Location of the base outfolder')
    args = parser.parse_args()
    return args

def get_mp3_and_transcript(c):
    '''
    Get the mp3 from case jsons
    '''
    js = json.load(open(c, encoding='utf-8'), object_pairs_hook=OrderedDict)

    rec_name = os.path.splitext(os.path.basename(c))[0]
    case_year = int(rec_name[:4])

    # Only want digital recordings, after October 2005
    if case_year <= 2004:
        return rec_name, False, False, None

    cutoff_date = dparser.parse('1 October 2005', fuzzy=True)

    if 'oral_argument_audio' in js:
        if js['oral_argument_audio']:
            if 'href' in js['oral_argument_audio']:
                t_url = js['oral_argument_audio']['href']
                resp = requests.get(t_url, timeout=20)
                if resp.ok:
                    js = json.loads(resp.content, object_pairs_hook=OrderedDict)
                else:
                    return rec_name, False, False, None

            elif len(js['oral_argument_audio']) == 1:
                if 'href' in js['oral_argument_audio'][0]:
                    t_url = js['oral_argument_audio'][0]['href']
                    resp = requests.get(t_url, timeout=20)
                    if resp.ok:
                        js = json.loads(resp.content, object_pairs_hook=OrderedDict)
                    else:
                        return rec_name, False, False, None

    if 'title' in js:
        dashind = js['title'].rfind('-')
        reduced_title = js['title'][dashind + 1:]
        start = reduced_title.find('(')
        end = reduced_title.find(')')
        if start != -1 and end != -1:
            reduced_title = reduced_title[:start - 1]
        try:
            case_date = dparser.parse(reduced_title, fuzzy=True)
            if case_date < cutoff_date:
                # print('Recording was too early')
                return rec_name, False, False, None
        except ValueError:
            # print('Couldnt figure out date {}'.format(reduced_title))
            return rec_name, False, js, None

        if 'media_file' in js:
            if 'transcript' in js:
                if js['media_file'] and js['transcript']:
                    for obj in js['media_file']:
                        if obj:
                            if 'href' in obj:
                                url = obj['href']
                                if url.endswith('mp3'):
                                    return rec_name, url, js, case_date

    return rec_name, False, js, None


def download_mp3(rec_name, url, outfile):
    filename = outfile
    if os.path.isfile(filename):
        return True
    try:
        request = urllib.request.urlopen(url, timeout=30)
        with open(filename+'_tmp', 'wb') as f:
             f.write(request.read())
        os.rename(filename+'_tmp', filename)
        return True
    except:
        os.remove(filename+'_tmp')
        return False


def filter_and_process_case(js,
                            valid_to_invalid_ratio=4,
                            min_num_utterances=5):
    '''
    This goes through and removes invalid utterances from a case json
    If the invalid utterances outnumber the valid ones by a ratio greater than
    valid_to_invalid_ratio, then False is returned
    else, the transcription is returned
    '''
    utts = []
    utts_spkr = []
    rec_speakers = {}

    invalid_utts = 0
    # Iterate through each speaker turn
    for sec in js['transcript']['sections']:
        for turn in sec['turns']:
            if not turn['speaker']:
                # ignore turns where no speaker is labelled
                invalid_utts += 1
                continue

            speaker = turn['speaker']
            speaker_id = speaker['identifier']

            if not speaker_id.strip():
                # ignore turns where no speaker is labelled
                invalid_utts += 1
                continue

            if speaker_id not in rec_speakers:
                rec_speakers[speaker_id] = speaker

            for utt in turn['text_blocks']:
                utt['speaker_id'] = speaker_id
                utt['start'] = float(utt['start'])
                utt['stop'] = float(utt['stop'])

                if utt['start'] >= utt['stop']:
                    invalid_utts += 1
                    continue
                else:
                    utts.append(utt)

    transcription = {'utts': utts, 'rec_speakers': rec_speakers}

    if len(utts) >= min_num_utterances:
        if invalid_utts > 0:
            if len(utts)/invalid_utts >= valid_to_invalid_ratio:
                return transcription
            else:
                return False
        else:
            return transcription
    else:
        return False

def process_case(c):
    '''
    Processes from the case json
    '''
    rec_name, url, js, case_date = get_mp3_and_transcript(c)
    if url: # If mp3 was found
        transcription = filter_and_process_case(js)
        if transcription: # If transcription is valid
            mp3_outfile = os.path.join(base_audio_folder, '{}.mp3'.format(rec_name))
            transcript_outfile = os.path.join(base_transcript_folder, '{}.json'.format(rec_name))

            transcription['case_date'] = case_date.strftime('%Y/%m/%d')
            download_success = download_mp3(rec_name, url, mp3_outfile)

            if download_success:
                with open(transcript_outfile, 'w', encoding='utf-8') as outfile:
                    json.dump(transcription, outfile)


def collate_speakers():
    '''
    Goes inside base_transcript_folder and collates all the speakers
    Writes to a dictionary of all speakers:

    speaker_id_dict = {speaker_id_key:{DICT}, etc.}

    This will be useful later on for parsing their full names, instead of the speakeaker_id_key
    '''
    transcripts = glob(os.path.join(base_transcript_folder, '*.json'))
    assert len(transcripts) > 1, 'Could only find 1 or less transcription json files'
    all_speaker_dict = {}

    for t in transcripts:
        js = json.load(open(t, encoding='utf-8'), object_pairs_hook=OrderedDict)
        for s in js['rec_speakers']:
            if s not in all_speaker_dict:
                all_speaker_dict[s] = js['rec_speakers'][s]

    outfile = os.path.join(base_outfolder, 'speaker_ids.json')
    with open(outfile, 'w', encoding='utf-8') as outfile:
        json.dump(all_speaker_dict, outfile)


def mp3_and_transcript_exist(c):
    mp3_exist = os.path.isfile(os.path.join(base_audio_folder, '{}.mp3'.format(os.path.splitext(os.path.basename(c))[0])))
    transcript_exist = os.path.isfile(os.path.join(base_transcript_folder, '{}.json'.format(os.path.splitext(os.path.basename(c))[0])))
    if mp3_exist and transcript_exist:
        return True
    else:
        False


if __name__ == '__main__':
    args = parse_args()
    case_folder = args.case_folder
    base_outfolder = args.base_outfolder
    os.makedirs(base_outfolder, exist_ok=True)

    base_audio_folder = os.path.join(base_outfolder, 'audio')
    os.makedirs(base_audio_folder, exist_ok=True)

    base_transcript_folder = os.path.join(base_outfolder, 'transcripts')
    os.makedirs(base_transcript_folder, exist_ok=True)

    cases = glob(os.path.join(case_folder, '20*'))
    print('{} cases found'.format(len(cases)))

    assert len(cases) > 1, "Could only find 1 or less case json files"

    trimmed_cases = [c for c in cases if not mp3_and_transcript_exist(c)]

    print('Processing {} cases...'.format(len(trimmed_cases)))

    with Pool(20) as p:
        for _ in tqdm(p.imap_unordered(process_case, trimmed_cases), total=len(trimmed_cases)):
            pass

    collate_speakers()

