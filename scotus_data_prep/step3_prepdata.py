import argparse
import datetime
import json
import os
import pickle
import sys
import shutil
from collections import OrderedDict
from copy import copy
from glob import glob

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Prep the data for verification, diarization and feature extraction')
    parser.add_argument('--base-outfolder', type=str, help='Location of the base outfolder')
    parser.add_argument('--subsegment-length', type=float, default=1.5, help='Length of diarization subsegments (default: 1.5s)')
    parser.add_argument('--subsegment-shift', type=float, default=0.75, help='Subsegment shift duration (default: 0.75s)')
    args = parser.parse_args()
    return args

def write_json(outfile, outdict):
    with open(outfile, 'w', encoding='utf-8') as wp:
        json.dump(outdict, wp)

def assign_newrecnames(caselist):
    """
    Converts recording names to a Kaldi friendly standard format of YEAR-XYZ

    input: recording json list
    output: (original to newrecname mapping, newrecname to original mapping)
    """
    recid_orec_mapping = OrderedDict({})

    for r in caselist:
        rec_name = os.path.splitext(os.path.basename(r))[0]
        year = rec_name[:4]
        assert len(year) == 4, 'Year length was not 4, something is wrong'
        index = 0
        new_recid = "{0}-{1:0=3d}".format(str(year), index)
        while True:
            if new_recid in recid_orec_mapping:
                index += 1
                new_recid = "{0}-{1:0=3d}".format(str(year), index)
            else:
                recid_orec_mapping[new_recid] = rec_name
                break

    orec_recid_mapping = OrderedDict({recid_orec_mapping[k]: k for k in recid_orec_mapping})
    return orec_recid_mapping, recid_orec_mapping


def make_wavscp(base_folder, caselist, orec_recid_mapping):
    """
    Make wav.scp with new recnames

    inputs: (outfile, list of case jsons, original to new recording mapping)
    outputs: None [file written to outfile]
    """
    wavlines = []
    wavscp = os.path.join(base_folder, 'wav.scp')
    for r in caselist:
        orec_name = os.path.splitext(os.path.basename(r))[0]
        newrec_id = orec_recid_mapping[orec_name]
        mp3file = os.path.join(os.path.abspath(base_folder), 'audio/{}.mp3'.format(orec_name))
        assert os.path.isfile(mp3file), "Couldn't find {}".format(mp3file)
        wavline = '{} ffmpeg -v 8 -i {} -f wav -ar 16000 -acodec pcm_s16le -|\n'.format(newrec_id, mp3file)
        wavlines.append(wavline)

    with open(wavscp, 'w+') as wp:
        for line in wavlines:
            wp.write(line)


def process_utts_dob(utts, dobs):
    '''
    Removes utterances from speakers with unknown dob
    '''
    new_utts = []
    for u in utts:
        if u['speaker_id'] in dobs:
            if dobs[u['speaker_id']]:
                new_utts.append(u)
    assert new_utts, "No more utts remain: this should not occur, check DoB pickle"
    return new_utts


def join_up_utterances(utt_list,
                       cutoff_dur=10000,
                       soft_min_length=0.0):
    new_utt_list = [utt_list[0].copy()]
    for i, utt in enumerate(utt_list):
        if i == 0:
            continue
        if utt['start'] == new_utt_list[-1]['stop'] and utt['speaker_id'] == new_utt_list[-1]['speaker_id']:
            if new_utt_list[-1]['stop'] - new_utt_list[-1]['start'] < cutoff_dur or utt_list[i]['stop'] - utt_list[i][
                'start'] < soft_min_length:
                new_utt_list[-1]['stop'] = utt['stop']
                new_utt_list[-1]['text'] += ' {}'.format(utt['text'])

        else:
            new_utt_list.append(utt_list[i].copy())
    return new_utt_list


def split_up_single_utterance(utt,
                              target_utt_length=10.0,
                              min_utt_length=4.0):
    duration = utt['stop'] - utt['start']
    if duration < min_utt_length + target_utt_length:
        return [utt]
    else:
        remaining_duration = copy(duration)
        new_utt_list = []
        iterations = 0
        while remaining_duration >= min_utt_length + target_utt_length:
            new_utt = OrderedDict({})
            new_utt['start'] = utt['start'] + (iterations * target_utt_length)
            new_utt['stop'] = new_utt['start'] + target_utt_length
            new_utt['text'] = utt['text']
            new_utt['speaker_id'] = utt['speaker_id']

            new_utt_list.append(new_utt)

            remaining_duration -= target_utt_length
            iterations += 1

        new_utt = OrderedDict({})
        new_utt['start'] = new_utt_list[-1]['stop']
        new_utt['stop'] = utt['stop']
        new_utt['text'] = utt['text']
        new_utt['speaker_id'] = utt['speaker_id']
        new_utt_list.append(new_utt)

        return new_utt_list


def split_up_long_utterances(utt_list,
                             target_utt_length=10.0,
                             min_utt_length=4.0):
    new_utt_list = []
    for utt in utt_list:
        splitup = split_up_single_utterance(utt,
                                            target_utt_length=target_utt_length,
                                            min_utt_length=min_utt_length)
        new_utt_list.extend(splitup)
    return new_utt_list


def make_segments(utts, recname, min_utt_len=2.0):
    """
    Make kaldi friendly segments of the format recname-VWXYZ

    Input: uttlist imported from transcription json, recname

    """
    seglines = []
    speakers = []
    utt_ids = []

    i = 0
    for utt in utts:
        utt_id = '{0}-{1:0=5d}'.format(recname, i)
        start = float(utt['start'])
        stop = float(utt['stop'])

        if float(start) + min_utt_len > float(stop):
            # Discard too short snippets
            continue

        speaker_id = utt['speaker_id']
        if not speaker_id.strip():
            # Discard empty speaker_ids
            continue

        line = '{} {} {} {}\n'.format(utt_id, recname, start, stop)

        seglines.append(line)
        speakers.append(speaker_id)
        utt_ids.append(utt_id)
        i += 1

    return seglines, utt_ids, speakers


def prep_utts(utts):
    # Converts start and stop to float
    for u in utts:
        u['start'] = float(u['start'])
        u['stop'] = float(u['stop'])
    return utts


def calculate_speaker_ages(speakers, dobs, rec_date):
    """
    Calculate the age in days of a list of speakers based on recording date and DoB

    inputs: (list of speakers, DoB dictionary of datetimes, recording datetime)
    output: list of speaker ages
    """
    set_speakers = set(speakers)
    age_dict = {}
    for s in set_speakers:
        dob = dobs[s]
        delta_days = abs((rec_date - dob).days)
        age_dict[s] = delta_days
    speaker_ages = [age_dict[s] for s in speakers]
    return speaker_ages


def make_verification_dataset(base_outfolder, caselist, orec_recid_mapping, dobs):
    """
    Makes a verification/training dataset: Long utterances split up
    """
    veri_data_path = os.path.join(base_outfolder, 'veri_data')
    os.makedirs(veri_data_path, exist_ok=True)
    wavscp_path = os.path.join(args.base_outfolder, 'wav.scp')
    shutil.copy(wavscp_path, veri_data_path)

    all_seglines = []
    all_uttids = []
    all_speakers = []
    all_recs = []
    all_ages = []

    for case in tqdm(caselist):
        rec_name = os.path.splitext(os.path.basename(case))[0]
        new_recid = orec_recid_mapping[rec_name]
        js = json.load(open(case, encoding='utf-8'), object_pairs_hook=OrderedDict)

        utts = js['utts']
        utts = process_utts_dob(utts, dobs)
        utts = prep_utts(utts)
        utts = join_up_utterances(utts)
        utts = split_up_long_utterances(utts, target_utt_length=10.0, min_utt_length=4.0)

        seglines, utt_ids, speakers = make_segments(utts, new_recid)

        case_date = datetime.datetime.strptime(js['case_date'], '%Y/%m/%d')
        utt_ages = calculate_speaker_ages(speakers, dobs, case_date)

        all_recs.extend([new_recid for _ in utt_ids])
        all_seglines.extend(seglines)
        all_uttids.extend(utt_ids)
        all_speakers.extend(speakers)
        all_ages.extend(utt_ages)

    with open(os.path.join(veri_data_path, 'segments'), 'w+') as fp:
        for line in all_seglines:
            fp.write(line)

    with open(os.path.join(veri_data_path, 'real_utt2spk'), 'w+') as fp:
        for u, s in zip(all_uttids, all_speakers):
            line = '{} {}\n'.format(u, s)
            fp.write(line)
    
    with open(os.path.join(veri_data_path, 'utt2age'), 'w+') as fp:
        for u, a in zip(all_uttids, all_ages):
            line = '{} {}\n'.format(u, a)
            fp.write(line)

    with open(os.path.join(veri_data_path, 'utt2spk'), 'w+') as fp:
        for u, r in zip(all_uttids, all_recs):
            line = '{} {}\n'.format(u, r)
            fp.write(line)

    utt2spk_to_spk2utt(os.path.join(veri_data_path, 'utt2spk'),
                       outfile=os.path.join(veri_data_path, 'spk2utt'))

    utt2spk_to_spk2utt(os.path.join(veri_data_path, 'real_utt2spk'),
                       outfile=os.path.join(veri_data_path, 'real_spk2utt'))



def split_up_single_utterance_subsegments(utt,
                              target_utt_length=1.5,
                              min_utt_length=1.4,
                              shift=0.75):
    """
    Split up a single utterance into subsegments based on input variables
    """
    duration = utt['stop'] - utt['start']
    if duration < min_utt_length + target_utt_length - shift:
        return [utt]
    else:
        new_utt_list = []
        current_start = copy(utt['start'])
        while current_start <= utt['stop'] - min_utt_length:
            new_utt = OrderedDict({})
            new_utt['start'] = current_start
            new_utt['stop'] = new_utt['start'] + target_utt_length
            new_utt['byte_start'] = utt['byte_start']  # todo fix
            new_utt['byte_stop'] = utt['byte_stop']  # todo fix
            new_utt['text'] = 'n/a'
            new_utt['speaker_id'] = utt['speaker_id']
            
            new_utt_list.append(new_utt)
            
            current_start += shift
        
        new_utt = OrderedDict({})
        new_utt['start'] = current_start
        new_utt['stop'] = utt['stop']
        new_utt['byte_start'] = utt['byte_start']
        new_utt['byte_stop'] = utt['byte_stop']
        new_utt['text'] = 'n/a'
        new_utt['speaker_id'] = utt['speaker_id']
        new_utt_list.append(new_utt)
        
        return new_utt_list


def split_up_uttlist_subsegments(utt_list,
                             target_utt_length=1.5,
                             min_utt_length=1.4,
                            shift=0.75):
    """
    Split up a list of utterances into subsegments based in input variables
    """
    new_utt_list = []
    for utt in utt_list:
        splitup = split_up_single_utterance_subsegments(utt,
                                            target_utt_length=target_utt_length,
                                            min_utt_length=min_utt_length,
                                                       shift=shift)
        new_utt_list.extend(splitup)
    return new_utt_list

def rttm_lines_from_uttlist(utts, rec_id):
    utts = join_up_utterances(utts)
    rttm_lines = []
    
    rttm_line = 'SPEAKER {} 0 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n'

    for u in utts:
        spk = u['speaker_id']
        start = float(u['start'])
        stop = float(u['stop'])
        offset = stop - start
        if offset < 0.0:
            continue
        rttm_lines.append(rttm_line.format(rec_id, start, offset, spk))
    
    return rttm_lines
        


def make_diarization_dataset(base_outfolder, caselist, orec_recid_mapping, dobs,
                            target_utt_length=1.5, min_utt_length=1.4, shift=0.75):
    """
    Makes a diarization dataset, utterances split into overlapping segments
    Makes rttms
    """
    diar_data_path = os.path.join(base_outfolder, 'diar_data')
    os.makedirs(diar_data_path, exist_ok=True)
    wavscp_path = os.path.join(args.base_outfolder, 'wav.scp')
    shutil.copy(wavscp_path, diar_data_path)

    all_seglines = []
    all_uttids = []
    all_speakers = []
    all_recs = []

    all_rttm_lines = []
    
    for case in tqdm(caselist):
        rec_name = os.path.splitext(os.path.basename(case))[0]
        new_recid = orec_recid_mapping[rec_name]
        js = json.load(open(case, encoding='utf-8'), object_pairs_hook=OrderedDict)

        utts = js['utts']
        utts = prep_utts(utts)
        utts = join_up_utterances(utts, cutoff_dur=1e20)

        rec_rttm_lines = rttm_lines_from_uttlist(utts, new_recid)
        all_rttm_lines.extend(rec_rttm_lines)

        utts = split_up_uttlist_subsegments(utts,
                                            target_utt_length=1.5,
                                            min_utt_length=1.4,
                                            shift=0.75)
        
        seglines, utt_ids, speakers = make_segments(utts, new_recid, min_utt_len=min_utt_length)

        all_recs.extend([new_recid for _ in utt_ids])
        all_seglines.extend(seglines)
        all_uttids.extend(utt_ids)
        all_speakers.extend(speakers)

    with open(os.path.join(diar_data_path, 'segments'), 'w+') as fp:
        for line in all_seglines:
            fp.write(line)
    
    with open(os.path.join(diar_data_path, 'ref.rttm'), 'w+') as fp:
        for line in all_rttm_lines:
            fp.write(line)

    with open(os.path.join(diar_data_path, 'real_utt2spk'), 'w+') as fp:
        for u, s in zip(all_uttids, all_speakers):
            line = '{} {}\n'.format(u, s)
            fp.write(line)

    with open(os.path.join(diar_data_path, 'utt2spk'), 'w+') as fp:
        for u, r in zip(all_uttids, all_recs):
            line = '{} {}\n'.format(u, r)
            fp.write(line)

    utt2spk_to_spk2utt(os.path.join(diar_data_path, 'utt2spk'),
                       outfile=os.path.join(diar_data_path, 'spk2utt'))

    utt2spk_to_spk2utt(os.path.join(diar_data_path, 'real_utt2spk'),
                       outfile=os.path.join(diar_data_path, 'real_spk2utt'))
        

def utt2spk_to_spk2utt(utt2spk_path, outfile=None):
    utts = []
    spks = []
    with open(utt2spk_path) as fp:
        for line in fp:
            splitup = line.strip().split(' ')
            assert len(splitup) == 2, 'Got more or less columns that was expected: (Got {}, expected 2)'.format(
                len(splitup))
            utts.append(splitup[0])
            spks.append(splitup[1])
    set_spks = sorted(list(set(spks)))
    spk2utt_dict = OrderedDict({k: [] for k in set_spks})
    for u, s in zip(utts, spks):
        spk2utt_dict[s].append(u)

    if outfile:
        with open(outfile, 'w+') as fp:
            for spk in spk2utt_dict:
                line = '{} {}\n'.format(spk, ' '.join(spk2utt_dict[spk]))
                fp.write(line)
    return spk2utt_dict



if __name__ == "__main__":
    args = parse_args()
    base_outfolder = args.base_outfolder
    assert os.path.isdir(base_outfolder), 'Outfolder does not exist'
    wavscp_path = os.path.join(args.base_outfolder, 'wav.scp')
    assert os.path.isfile(wavscp_path),  "Can't find {}".format(wavscp_path)
    
    dobs_pkl_path = os.path.join(args.base_outfolder, 'dob.p')
    assert os.path.isfile(dobs_pkl_path), "Couldn't find {}".format(dobs_pkl_path)
    dobs = pickle.load(open(dobs_pkl_path, 'rb'))

    audio_folder = os.path.join(base_outfolder, 'audio')
    transcript_folder = os.path.join(base_outfolder, 'transcripts')

    caselist = glob(os.path.join(transcript_folder, '*.json'))
    caselist = sorted(caselist)

    print('Assigning new recording names...')
    orec_recid_mapping, recid_orec_mapping = assign_newrecnames(caselist)

    write_json(os.path.join(base_outfolder, 'orec_recid_mapping.json'), orec_recid_mapping)
    write_json(os.path.join(base_outfolder, 'recid_orec_mapping.json'), recid_orec_mapping)

    print('Making wav.scp in {}'.format(os.path.join(base_outfolder, 'wav.scp')))
    make_wavscp(base_outfolder, caselist, orec_recid_mapping)

    print('Making base verification/training dataset...')
    make_verification_dataset(base_outfolder, caselist, orec_recid_mapping, dobs)

    print('Making diarization dataset...')
    make_diarization_dataset(base_outfolder, caselist, orec_recid_mapping, dobs,
                                target_utt_length=args.subsegment_length, min_utt_length=args.subsegment_length-0.1, 
                                shift=args.subsegment_shift)

