import eeg
import matplotlib.pyplot as plt

optpath = ('/export/home/mike/matlab/batch/nki_err_bssnew_CSD/'
           'nki_err_cnth1s_bssnew_CSD_opt.mat')

csvpath = '/export/home/mike/matlab/csv/nki.csv'

r = eeg.Results(optpath, csvpath, 15)

''' testing plot_erp '''
r.plot_erp(figure_by={'channel': ['FZ', 'CZ', 'PZ']},
           subplot_by={'POP': None},
           glyph_by={'condition': None}
           )

# testing by-arg permutations
r.plot_erp(figure_by={'condition': None},
           subplot_by={'POP': None},
           glyph_by={'channel': ['FZ', 'CZ', 'PZ']}
           )

r.plot_erp(figure_by={'POP': None},
           subplot_by={'channel': ['FZ', 'CZ', 'PZ']},
           glyph_by={'condition': None}
           )

# testing condition differences
r.plot_erp(figure_by={'channel': ['FZ', 'CZ', 'PZ']},
           subplot_by={'POP': None},
           glyph_by={'condition': ['P', 'N', {'minus': ['P', 'N']}]}
           )

''' testing tf plotting '''
# with power
r.plot_tf(measure='power',
          figure_by={'POP': None, 'channel': ['FZ']},
          subplot_by={'condition': None}
          )

# condition difference
r.plot_tf(measure='power',
          figure_by={'channel': ['FZ']},
          subplot_by={'POP': None, 'condition': [{'minus': ['P', 'N']}]},
          )

# change measure to itc
r.plot_tf(measure='itc',
          figure_by={'POP': None, 'channel': ['FZ']},
          subplot_by={'condition': None},
          lims='minmax'
          )

# use a different cmap
r.plot_tf(measure='itc',
          figure_by={'channel': ['FZ']},
          subplot_by={'POP': None, 'condition': [{'minus': ['P', 'N']}]},
          cmap_override=plt.cm.RdBu_r
          )

# change measure to coh
r.plot_tf(measure='coh',
          figure_by={'POP': None, 'pair': ['F4~F3']},
          subplot_by={'condition': None},
          lims='minmax',
          )

# use a different cmap
r.plot_tf(measure='coh',
          figure_by={'pair': ['F4~F3']},
          subplot_by={'POP': None, 'condition': [{'minus': ['P', 'N']}]},
          cmap_override=plt.cm.RdBu_r
          )

''' testing plot_topo '''
# with erp
r.plot_topo(measure='erp',
            times=list(range(0, 501, 50)),
            figure_by={'POP': ['C']},
            row_by={'condition': None}
            )

# condition difference
r.plot_topo(measure='erp',
            times=list(range(0, 501, 50)),
            figure_by={'POP': ['C']},
            row_by={'condition': [{'minus': ['P', 'N']}]},
            )

# with power
r.plot_topo(measure='power',
            times=list(range(0, 501, 125)),
            figure_by={'POP': ['C'], 'frequency': [[4, 7]]},
            row_by={'condition': None}
            )

# condition difference
r.plot_topo(measure='power',
            times=list(range(0, 501, 125)),
            figure_by={'POP': ['C'], 'frequency': [[4, 7]]},
            row_by={'condition': [{'minus': ['P', 'N']}]},
            )

# with itc
r.plot_topo(measure='itc',
            times=list(range(0, 501, 125)),
            figure_by={'POP': ['C'], 'frequency': [[4, 7]]},
            row_by={'condition': None},
            lims='minmax'
            )

# condition difference
r.plot_topo(measure='itc',
            times=list(range(0, 501, 125)),
            figure_by={'POP': ['C'], 'frequency': [[4, 7]]},
            row_by={'condition': [{'minus': ['P', 'N']}]},
            cmap_override=plt.cm.RdBu_r
            )

''' testing plot_arctopo '''
r.plot_arctopo(times=list(range(0, 501, 125)),
               figure_by={'POP': ['C'], 'frequency': [[4, 7]]},
               row_by={'condition': None},
               lims='minmax'
               )

# condition difference
r.plot_arctopo(times=list(range(0, 501, 125)),
               figure_by={'POP': ['C'], 'frequency': [[4, 7]]},
               row_by={'condition': [{'minus': ['P', 'N']}]},
               cmap_override=plt.cm.RdBu_r
               )
