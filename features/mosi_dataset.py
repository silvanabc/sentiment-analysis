################################################
# Extract features from MOSI Dataset
################################################


#Imports
import sys
import argparse
import features.extraction as extract
import datetime
import numpy as np
import pandas as pd
import pickle

# train, validation and test videos
standard_train_fold=['2iD-tVS8NPw', '8d-gEyoeBzc', 'Qr1Ca94K55A', 'Ci-AH39fi3Y', '8qrpnFRGt2A', 'Bfr499ggo-0', 'QN9ZIUWUXsY', '9T9Hf74oK10', '7JsX8y1ysxY', '1iG0909rllw', 'Oz06ZWiO20M', 'BioHAh1qJAQ', '9c67fiY0wGQ', 'Iu2PFX3z_1s', 'Nzq88NnDkEk', 'Clx4VXItLTE', '9J25DZhivz8', 'Af8D0E4ZXaw', 'TvyZBvOMOTc', 'W8NXH0Djyww', '8OtFthrtaJM', '0h-zjBukYpk', 'Vj1wYRQjB-o', 'GWuJjcEuzt8', 'BI97DNYfe5I', 'PZ-lDQFboO8', '1DmNV9C1hbY', 'OQvJTdtJ2H4', 'I5y0__X72p0', '9qR7uwkblbs', 'G6GlGvlkxAQ', '6_0THN4chvY', 'Njd1F0vZSm4', 'BvYR0L6f2Ig', '03bSnISJMiM', 'Dg_0XKD0Mf4', '5W7Z1C_fDaE', 'VbQk4H8hgr0', 'G-xst2euQUc', 'MLal-t_vJPM', 'BXuRRbG0Ugk', 'LSi-o-IrDMs', 'Jkswaaud0hk', '2WGyTLYerpo', '6Egk_28TtTM', 'Sqr0AcuoNnk', 'POKffnXeBds', '73jzhE8R1TQ', 'OtBXNcAL_lE', 'HEsqda8_d0Q', 'VCslbP0mgZI', 'IumbAb8q2dM']

standard_valid_fold=['WKA5OygbEKI', 'c5xsKMxpXnc', 'atnd_PF-Lbs', 'bvLlb-M3UXU', 'bOL9jKpeJRs', '_dI--eQ6qVU', 'ZAIRrfG22O0', 'X3j2zQgwYgE', 'aiEXnCPZubE', 'ZUXBRvtny7o']

standard_test_fold=['tmZoasNr4rU', 'zhpQhgha_KU', 'lXPQBPVc5Cw', 'iiK8YX8oH1E', 'tStelxIAHjw', 'nzpVDcQ0ywM', 'etzxEpPuc6I', 'cW1FSBF59ik', 'd6hH302o4v8', 'k5Y_838nuGo', 'pLTX3ipuDJI', 'jUzDDGyPkXU', 'f_pcplsH_V0', 'yvsjCA6Y5Fc', 'nbWiPyCm4g0', 'rnaNMUZpvvg', 'wMbj6ajWbic', 'cM3Yna7AavY', 'yDtzw_Y-7RU', 'vyB00TXsimI', 'dq3Nf_lMPnE', 'phBUpBr1hSo', 'd3_k5Xpfmik', 'v0zCBqDeKcE', 'tIrG4oNLFzE', 'fvVhgmXxadc', 'ob23OKe5a9Q', 'cXypl4FnoZo', 'vvZ4IcEtiZc', 'f9O3YtZ2VfI', 'c7UH_rxdZv4']

TEST_FEATURES_PATH = './output/mosi_train.npy'
TRAIN_FEATURES_PATH = './output/mosi_test.npy'


def get_labels_and_length(filenames, videos_path, sep, max_utterance, df):
    info_names, _ = extract.get_video_info(videos_path, sep)

    labels = np.empty((0, max_utterance))
    utterance_length = []

    for name in sorted(info_names):
        if (name in filenames):
            df_segment = df[df.video_id == name].sort_values(by=['segment'])
            scores = df_segment['score'].values
            labels = np.append(labels, [extract.pad_array(scores, max_utterance)], axis=0)

            utterance_length.append(df_segment.shape[0])

    return labels, utterance_length


def generate_pickle(csv_path, videos_path, sep='_',
                    train_features_path=TRAIN_FEATURES_PATH,
                    test_features_path=TEST_FEATURES_PATH,
                    pickle_path='mosi_pickle.pkl'):
    # format:
    #(train_data, train_label, test_data, test_label, maxlen, train_length, test_length)
    df = pd.read_csv(csv_path, skiprows=0, names=['start', 'end', 'video_id', 'segment', 'score'])

    train = np.load(train_features_path)
    test = np.load(test_features_path)

    max_utterance = train.shape[1]

    train_fold = standard_train_fold + standard_valid_fold
    train_label, train_length = get_labels_and_length(train_fold, videos_path, sep, max_utterance, df)

    test_folder = standard_test_fold
    test_label, test_length = get_labels_and_length(test_folder, videos_path, sep, max_utterance, df)

    data = train, train_label, test, test_label, train.shape[1], train_length, test_length

    pickle_out = open(pickle_path, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

    print('Pickle generated')


def parse_arguments():
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, nargs='?', const=True)
    parser.add_argument("--output_path", type=str, nargs='?', const=True, default='./output/')
    parser.add_argument("--output_name", type=str, nargs='?', const=True)
    parser.add_argument("--start_segment", type=int, nargs='?', const=True, default=1)
    parser.add_argument("--sep_segment", type=str, nargs='?', const=True, default='_')
    parser.add_argument("--label_csv_path", type=str, nargs='?', const=True, default='./OpinionLevelSentiment.csv')
    args, _ = parser.parse_known_args(argv)

    if not args.path:
        raise argparse.ArgumentError('Expected value for path')

    if not args.output_name:
        now = datetime.datetime.now()
        args.output_name = str(int(now.timestamp()))

    return args


def extract_video_features(video_names, args):
    video_features = extract.get_video_features(args.path, args.sep_segment, args.start_segment, video_names)
    output_path = args.output_path + args.output_name
    np.save(output_path, video_features)
    print('\nVideo features saved in', output_path)
    return output_path

if __name__ == "__main__":
    args = parse_arguments()

    f = ['2iD-tVS8NPw', '8d-gEyoeBzc', 'Qr1Ca94K55A']

    # args.output_name = 'test_'
    # s = extract_video_features(standard_test_fold, args)
    # print(s)

    #
    # #-- Test --#
    # args.output_name = 'test_'
    # test_features_path = extract_video_features(standard_test_fold, args)
    #
    # # -- Train --#
    # args.output_name = 'train_'
    # train_features_path = extract_video_features(standard_valid_fold + standard_train_fold, args)
    #
    # #-- Create pickle --#
    # pklname = 'mosi_video'
    train_features_path = './output/train_.npy'
    test_features_path = './output/test_.npy'
    generate_pickle(args.label_csv_path, args.path, args.sep_segment, train_features_path, test_features_path)
