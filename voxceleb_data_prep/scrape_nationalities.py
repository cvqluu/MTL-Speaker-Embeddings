from collections import Counter, OrderedDict
from pprint import pprint

import pandas as pd
import spacy
import wikipedia
import wptools
from tqdm import tqdm
from wikipedia import DisambiguationError


def wiki_infobox(text):
    try:
        page = wptools.page(text, silent=True).get_parse()
        infobox = page.data['infobox']
    except:
        infobox = {}
    return infobox


if __name__ == "__main__":
    vox1 = pd.read_csv('./data/vox1_meta.csv', delimiter='\t')
    vox2 = pd.read_csv('./data/vox2_meta.csv')
    vgg2 = pd.read_csv('./data/vggface2_meta.csv', quotechar='"', skipinitialspace=True)
    us_states = set(pd.read_csv('./data/us_states.csv')['States'].str.lower().values)

    vgg_id_to_name = {k:v.strip() for k,v in zip(vgg2['Class_ID'].values, vgg2['Name'].values)}
    vox2_ids_dict = {k:v.strip() for k,v in zip(vox2['VoxCeleb2 ID '].values, vox2['VGGFace2 ID '])}
    vox2_id_to_name = {k:vgg_id_to_name[vox2_ids_dict[k]] for k in vox2_ids_dict}
    vox2_name_to_id = {k:v for v, k in vox2_id_to_name.items()}

    natcountry = pd.read_csv('./data/nationality_to_country.tsv', delimiter='\t')
    country_set = set(natcountry.Country.values)
    country_nat_dict = {k.lower():[] for k in country_set}

    for c, n in zip(natcountry.Country.values, natcountry.Nationality.values):
        country_nat_dict[c.lower()].append(n.lower())
        
    nat_country_dict = {n.lower():c.lower() for c, n in zip(natcountry.Country.values, natcountry.Nationality.values)}

	# Reorder country nat_dict based on demographics of vox1
	# Most populous first
	# Then by length of country name
	# This is to make sure the country names which are substrings are not checked first
	common_nats = vox1.Nationality.str.lower().value_counts().keys()

	common_nats_keys = []
	for c in common_nats:
		common_nats_keys.append(nat_country_dict[c])

	ordered_country_nat_dict = OrderedDict({})
	for c in common_nats_keys:
		ordered_country_nat_dict[c] = country_nat_dict[c]

	countries_by_len = sorted(country_nat_dict.keys(), key=len, reverse=True)    

	for c in countries_by_len:
		if c not in ordered_country_nat_dict:
			ordered_country_nat_dict[c] = country_nat_dict[c]

    vox2_names = list(vox2_id_to_name.values())
    vox2_nationalities = OrderedDict({k:[] for k in vox2_names})

    nlp = spacy.load("en_core_web_sm")

    for i, name in enumerate(tqdm(vox2_nationalities)):
        if vox2_nationalities[name]:
            continue
            
        # Get the wikipedia page and summary text
        qname = ' '.join(name.split('_'))
        try:
            text = wikipedia.summary(qname)
        except:
            search = wikipedia.search(qname, results=3)
            if len(search) == 0:
                print(name)
                continue
            else:
                index = 0
                while True:
                    if index >= len(search):
                        qname = 'nan'
                        text = ''
                        break
                    try:
                        qname = search[index]
                        text = wikipedia.summary(qname, auto_suggest=False)
                        break
                    except DisambiguationError:
                        index += 1
        
        if qname == 'nan':
            #Couldn't find a good wikipedia page
            print(name)
            continue
        
        #Try the infobox first
        try:
            person_infobox = wiki_infobox(qname)
            if 'birth_place' in person_infobox:
                place = person_infobox['birth_place'].replace('[', '').replace(']', '').lower()
                
                for s in us_states:
                    if s in place:
                        vox2_nationalities[name] = ['united states']
                        break

                if vox2_nationalities[name]:
                    continue
                
                for c in ordered_country_nat_dict:
                    if c in place:
                        vox2_nationalities[name] = [c]
                        break

                if vox2_nationalities[name]:
                    continue

                place_infobox = wiki_infobox(place)
                if 'subdivision_name' in place_infobox:
                    subd_country = place_infobox['subdivision_name'].lower()
                    if subd_country in country_nat_dict:
                        vox2_nationalities[name] = [subd_country]
                        continue
        except:
            pass
                    
        
        #Otherwise try the summary text
        doc = nlp(text)
        all_stopwords = nlp.Defaults.stop_words
        
        nat_candidates = []
        for j, tok in enumerate(doc):
            if tok.text.lower() in nat_country_dict:
                if doc[j-1].text.lower() == 'new':
                    continue
                if tok.text.lower() == 'american' and doc[j+1].text.lower() == 'football':
                    continue
                nat_candidates.append(nat_country_dict[tok.text.lower()])
                if doc[j+1].text.lower() == '-':
                    if doc[j+2].text.lower() == 'born':
                        c = doc[j+3].text.lower()
                        if c in nat_country_dict:
                            nat_candidates = [nat_country_dict[c]]
                            break
        vox2_nationalities[name] = nat_candidates

    vox2_nats_final = OrderedDict({})
    for name in vox2_nationalities:
        # Take the most frequent nat in the list
        # In case of ties, takes the first occuring one
        if vox2_nationalities[name]:
            best_nat = Counter(vox2_nationalities[name]).most_common(1)[0][0]
            vox2_nats_final[name] = best_nat
        else:
            vox2_nats_final[name] = 'unk'


    with open('./spk2nat', 'w') as fp:
        for name in vox2_nats_final:
            id = vox2_name_to_id[name]
            nat = '_'.join(vox2_nats_final[name].split()) 
            line = '{} {}\n'.format(id, nat)
            fp.write(line)

