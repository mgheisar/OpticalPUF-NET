import datetime
import sys
if __name__ == '__main__':
    # in_filename = '/home/mgheisar/Music/If.Beale.Street.Could.Talk-2018.srt'
    # out_filename = '/home/mgheisar/Music/If.Beale.Street.Could.Talk-2018-shifted.srt'
    # shift = +15  ## sec

    in_filename = sys.argv[1]
    out_filename = sys.argv[2]
    shift = float(sys.argv[3])

    out_file = open(out_filename, 'w')
    with open(in_filename) as in_file:
        Lines = in_file.readlines()
        for line_str in Lines:
            if '-->' in line_str:
                segments = line_str.split(' --> ')
                start_time = datetime.datetime.strptime(segments[0], '%H:%M:%S,%f')
                end_time = datetime.datetime.strptime(segments[1].replace('\n', ''), '%H:%M:%S,%f')

                start_time_new = start_time + datetime.timedelta(seconds=shift)
                end_time_new = end_time + datetime.timedelta(seconds=shift)

                line_str = start_time_new.strftime('%H:%M:%S,%f') + ' --> ' + \
                    end_time_new.strftime('%H:%M:%S,%f') + '\n'
                print(line_str)
            out_file.write(line_str)
    out_file.close()
