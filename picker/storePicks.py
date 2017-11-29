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


def save_pdf():
    filename = os.path.split(save_data['avgh1 path'])[1]
    with PdfPages(os.path.join(save_data['save dir'], filename + '.pdf')) as pdf:
        output_page(save_data['plot desc'][:8])
        pdf.savefig()
        plt.close()
        output_page(save_data['plot desc'][8:])
        pdf.savefig()
        plt.close()

def output_page(layout_desc):
    # setup
    xlim = [-100, 850]
    xticks = [0, 250, 500, 750]
    tick_col = 0
    ylim = [-4, 12]
    yticks = [0, 10]
    arrow_size = 4.5
    linewidth = 0.5

    ccns = [[v / 255 for v in cc] for cc in save_data['plot props']['line colors']]

    fig = plt.figure(figsize=(11, 8.5))
    nrows = len(layout_desc)
    ncols = len(layout_desc[0])
    spNum = 0
    for rN, prow in enumerate(layout_desc):

        for cN, p_desc in enumerate(prow):
            spNum += 1
            if p_desc:
                elec = p_desc['electrode']

                #    plot = s.plotsGrid.addPlot(rN + 1, cN)  # ,title=elec)
                ax = plt.subplot(nrows + 1, ncols, spNum)
                plt.subplots_adjust(hspace=0.001, wspace=0.001)
                ax.margins(0)
                ax.set_xlim(xlim)
                ax.set_xticks(xticks)
                ax.set_ylim(ylim)
                ax.set_yticks(yticks)
                ax.tick_params(direction='out', pad=5, labelsize=7)

                ax.add_artist(plt.Line2D((xticks[0], xticks[-1]), (yticks[0], yticks[0]),
                                         color='black', linewidth=0.5))
                ax.add_artist(plt.Line2D((0, 0), yticks, color='black', linewidth=0.5))
                if rN == nrows - 1:
                    ax.set_xticklabels(xticks, fontsize=6)
                else:
                    ax.set_xticklabels([])
                if cN == tick_col:
                    ax.set_yticklabels(yticks, fontsize=6)
                else:
                    ax.set_yticklabels([])

                ax.set_ylabel(elec, rotation=0, fontsize=10, labelpad=5)
                ax.set_frame_on(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()
                ax.spines['left'].set_position('zero')
                ax.spines['bottom'].set_position('zero')
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                for caseN, case in enumerate(save_data['experiment cases']):
                    if case in save_data['working cases']:
                        workingN = save_data['working cases'].index(case)
                        # case_color = s.plot_props['line colors'][caseN]
                        ccn = ccns[workingN]  # [v / 255 for v in case_color]
                        ax.plot(save_data['current data']['times'],
                                save_data['current data'][elec + '_' + case],
                                color=ccn, clip_on=False,
                                linewidth=linewidth)

                        peak_keys = [k for k in save_data['peak data'].keys() if k[0] == elec and k[1] == case]
                        for pk in peak_keys:
                            if pk[2][0] == 'P':
                                arrow_len = arrow_size
                            else:
                                arrow_len = -arrow_size
                            amp, lat = save_data['peak data'][pk]
                            ax.annotate('', (lat, amp), (lat, amp + arrow_len),
                                        size=7, clip_on=False, annotation_clip=False,
                                        arrowprops=dict(arrowstyle='-|>',
                                                        fc=ccn, ec=ccn))
    # info row on bottom
    file_info = save_data['info']
    subD = eeg.subject_data()
    expD = eeg.exp_data()
    tformD = eeg.transforms_data()
    casesD = eeg.case_data()
    runD = eeg.run_data()
    # print('file_info',file_info)
    # print('subject data',subD)
    # print('exp data', expD)
    # print('transform data', tformD)
    # print('run data',runD)
    # print( 'eeg cases',casesD )


    filename = os.path.split(save_data['avgh1 path'])[1]
    desc_ax = plt.subplot(nrows + 1, ncols, spNum + 1)
    desc_ax.set_frame_on(False)
    desc_ax.set_xticks([])
    desc_ax.set_yticks([])
    desc_ax.set_xlim([0, 10])
    desc_ax.set_ylim([0, 10])
    desc_ax.text(0, 0, filename + '\n' + \
                 ' '.join([str(round(subD['age'] * 100) / 100)[:5], ' ', subD['gender'],
                           ' ', subD['handedness'], '  ', 'artf thresh',
                           str(expD['threshold_value'])]) + ' uV \n' + \
                 runD['run_date_time'],
                 fontsize=9)

    trials_table_rows_ax = plt.subplot(nrows + 1, ncols, spNum + 3)
    trials_table_rows_ax.set_frame_on(False)
    trials_table_rows_ax.set_xticks([])
    trials_table_rows_ax.set_yticks([])
    trials_table_rows_ax.set_ylim([0, 10])
    trials_table_rows_ax.set_xlim([0, 10])

    trials_table_ax = plt.subplot(nrows + 1, ncols, spNum + 4)
    trials_table_ax.set_frame_on(False)
    trials_table_ax.set_xticks([])
    trials_table_ax.set_yticks([])
    trials_table_ax.set_ylim([0, 10])
    trials_table_ax.text(0, 8, 'trials     resps', fontsize=9)

    rx_time_ax = plt.subplot(nrows + 1, ncols, spNum + 5)
    rx_time_ax.set_ylim([0, 1])
    rx_time_ax.text(0, 0.8, 'Response Times', fontsize=8)
    rx_time_ax.set_frame_on(False)
    rx_time_ax.set_xticks(xticks)
    rx_time_ax.set_xticklabels(xticks, fontsize=6)
    rx_time_ax.xaxis.set_ticks_position('bottom')
    rx_time_ax.set_yticks([])
    rx_time_ax.set_xlim(xlim)
    rx_time_ax.add_artist(plt.Line2D((xticks[0], xticks[-1]), (yticks[0], yticks[0]),
                                     color='black', linewidth=0.5))
    for caseN, case in enumerate(save_data['working cases']):
        case_alias = save_data['case aliases'][case]
        ccn = ccns[caseN]
        cD = casesD[case]
        trials_table_rows_ax.text(9, 6 - 2 * caseN, cD['descriptor'],
                                  horizontalalignment='right',
                                  fontsize=9, color=ccn)
        n_trials = str(cD['n_trials_accepted'])
        n_resp = str(min([cD['n_responses'], cD['n_trials_accepted']]))
        trials_table_ax.text(0, 6 - 2 * caseN, ' ' + n_trials + \
                             ' ' * (3 - len(n_trials)) + '       ' \
                             + n_resp,
                             fontsize=9, color=ccn)

        rx_tm = cD['mean_resp_time']
        rx_time_ax.plot(2 * [rx_tm], [0.15, 0.5], color=ccn)
        rx_time_ax.text(rx_tm, 0.55, str(round(rx_tm * 100) / 100), fontsize=7, color=ccn)

    fig.tight_layout()

print('Saving ...')
save_mt()
save_pdf()
print('Finished background save')