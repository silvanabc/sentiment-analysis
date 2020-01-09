################################################
# Extract features from MOSI Dataset
################################################


#Imports
import sys
import argparse
import features.extraction as extract
import datetime
import numpy as np

# train, validation and test videos
standard_train_fold=['2iD-tVS8NPw', '8d-gEyoeBzc', 'Qr1Ca94K55A', 'Ci-AH39fi3Y', '8qrpnFRGt2A', 'Bfr499ggo-0', 'QN9ZIUWUXsY', '9T9Hf74oK10', '7JsX8y1ysxY', '1iG0909rllw', 'Oz06ZWiO20M', 'BioHAh1qJAQ', '9c67fiY0wGQ', 'Iu2PFX3z_1s', 'Nzq88NnDkEk', 'Clx4VXItLTE', '9J25DZhivz8', 'Af8D0E4ZXaw', 'TvyZBvOMOTc', 'W8NXH0Djyww', '8OtFthrtaJM', '0h-zjBukYpk', 'Vj1wYRQjB-o', 'GWuJjcEuzt8', 'BI97DNYfe5I', 'PZ-lDQFboO8', '1DmNV9C1hbY', 'OQvJTdtJ2H4', 'I5y0__X72p0', '9qR7uwkblbs', 'G6GlGvlkxAQ', '6_0THN4chvY', 'Njd1F0vZSm4', 'BvYR0L6f2Ig', '03bSnISJMiM', 'Dg_0XKD0Mf4', '5W7Z1C_fDaE', 'VbQk4H8hgr0', 'G-xst2euQUc', 'MLal-t_vJPM', 'BXuRRbG0Ugk', 'LSi-o-IrDMs', 'Jkswaaud0hk', '2WGyTLYerpo', '6Egk_28TtTM', 'Sqr0AcuoNnk', 'POKffnXeBds', '73jzhE8R1TQ', 'OtBXNcAL_lE', 'HEsqda8_d0Q', 'VCslbP0mgZI', 'IumbAb8q2dM']

standard_valid_fold=['WKA5OygbEKI', 'c5xsKMxpXnc', 'atnd_PF-Lbs', 'bvLlb-M3UXU', 'bOL9jKpeJRs', '_dI--eQ6qVU', 'ZAIRrfG22O0', 'X3j2zQgwYgE', 'aiEXnCPZubE', 'ZUXBRvtny7o']

standard_test_fold=['tmZoasNr4rU', 'zhpQhgha_KU', 'lXPQBPVc5Cw', 'iiK8YX8oH1E', 'tStelxIAHjw', 'nzpVDcQ0ywM', 'etzxEpPuc6I', 'cW1FSBF59ik', 'd6hH302o4v8', 'k5Y_838nuGo', 'pLTX3ipuDJI', 'jUzDDGyPkXU', 'f_pcplsH_V0', 'yvsjCA6Y5Fc', 'nbWiPyCm4g0', 'rnaNMUZpvvg', 'wMbj6ajWbic', 'cM3Yna7AavY', 'yDtzw_Y-7RU', 'vyB00TXsimI', 'dq3Nf_lMPnE', 'phBUpBr1hSo', 'd3_k5Xpfmik', 'v0zCBqDeKcE', 'tIrG4oNLFzE', 'fvVhgmXxadc', 'ob23OKe5a9Q', 'cXypl4FnoZo', 'vvZ4IcEtiZc', 'f9O3YtZ2VfI', 'c7UH_rxdZv4']


def create_picke():
    # format:
    #(train_data, train_label, test_data, test_label, maxlen, train_length, test_length)
    return

def parse_arguments():
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, nargs='?', const=True)
    parser.add_argument("--output_path", type=str, nargs='?', const=True, default='./output/')
    parser.add_argument("--output_name", type=str, nargs='?', const=True)
    parser.add_argument("--start_segment", type=int, nargs='?', const=True, default=1)
    parser.add_argument("--sep_segment", type=str, nargs='?', const=True, default='_')
    args, _ = parser.parse_known_args(argv)

    if not args.path:
        raise argparse.ArgumentError('Expected value for path')

    if not args.output_name:
        now = datetime.datetime.now()
        args.output_name = str(int(now.timestamp()))

    return args


if __name__ == "__main__":
    args = parse_arguments()

    f = ['2iD-tVS8NPw', '8d-gEyoeBzc']
    # video_features = extract.get_video_features(args.path, args.sep_segment, args.start_segment, f)
    # output_path = args.output_path + args.output_name
    # np.save(output_path, video_features)
    # print('\nVideo features saved in', output_path)


    #-- Test --#

    list = []
    count = 0
    start_loop_test_time = datetime.datetime.now()
    for video_id in standard_test_fold:
        start_time = datetime.datetime.now()
        count += 1
        print(100 * '-')
        print('Video', count)
        print(100 * '-')

        video_features = extract.get_video_features(args.path, args.sep_segment, args.start_segment, [video_id])

        list.append(video_features)

        print("Computation Time: ", str(datetime.datetime.now() - start_time))

    print("Total Computation Time for Test Folder: ", str(datetime.datetime.now() - start_loop_test_time))

    output_path = args.output_path + args.output_name + "_test"
    np.save(output_path, np.array(video_features))

    print('\nVideo features saved in', output_path)

    # # -- Train --#
    # train = standard_train_fold + standard_valid_fold
    # video_features = extract.get_video_features(args.path, args.sep_segment, args.start_segment, train)
    # output_path = args.output_path + args.output_name + "_train"
    # np.save(output_path, video_features)
    # print('\nVideo features saved in', output_path)
