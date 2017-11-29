import sys, os, pickle
import numpy as np
from picker.EEGdata import avgh1

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#from picker.PeakPickerQ import Picker

data_path = sys.argv[1]

with open(data_path,'rb') as rf:
    save_data = pickle.load(rf)

print('save_data keys:', [k for k in save_data.keys()])

eeg = avgh1(save_data['avgh1 path'])
eeg.extract_case_data()
eeg.extract_mt_data()

def save_mt( ):
    ''' saves pickled picks as an HBNL-formatted *.mt text file '''

    #s.debug('save_mt',1)
    picked_cp = save_data['picks']
    cases_Ns = list(set([(cp[0], eeg.case_num_map[cp[0]]) for cp in picked_cp]))
    peaks = list(set([cp[1] for cp in picked_cp]))

    n_cases = len(cases_Ns)
    n_chans = 61  # only core 61 chans
    n_peaks = len(peaks)
    amps = np.empty((n_peaks, n_chans, n_cases))
    lats = np.empty((n_peaks, n_chans, n_cases))
    amps.fill(np.NAN)
    lats.fill(np.NAN)
    for icase, case_N in enumerate(cases_Ns):
        case_name = case_N[0]
        for ichan, chan in enumerate( eeg.electrodes_61 ):  # only core 61 chans
            case_peaks = [cp[1] for cp in picked_cp if cp[0] == case_name]
            case_peaks.sort()
            for peak in case_peaks:
                ipeak = peaks.index(peak)
                pd_key = (chan, case_name, peak)
                if pd_key in save_data['peak data']:
                    amp_lat = save_data['peak data'][(chan, case_name, peak)]
                    amps[ipeak, ichan, icase] = amp_lat[0]
                    lats[ipeak, ichan, icase] = amp_lat[1]
                else:
                    print(['Missing peak data for:',pd_key],3)

    # reshape into 1d arrays
    amps1d = amps.ravel('F')
    lats1d = lats.ravel('F')

    # build mt text (makes default output location), write to a test location
    eeg.build_mt([cn[1] for cn in cases_Ns], peaks, amps1d, lats1d)

    if not os.path.exists(save_data['save dir']):
        os.mkdir(save_data['save dir'])
    fullpath = os.path.join( save_data['save dir'], eeg.mt_name)
    of = open(fullpath, 'w')
    of.write(eeg.mt)
    of.close()
    print('Saved ', fullpath)
    #s.status_message(text='Saved to ' + os.path.split(fullpath)[0])
    #s.debug(['Saved', fullpath],1)

print('Saving ...')
save_mt()