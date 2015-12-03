from bokeh.plot_object import PlotObject
from bokeh.server.utils.plugins import object_page
from bokeh.server.app import bokeh_app
from bokeh.plotting import curdoc, cursession
from bokeh.crossfilter.models import CrossFilter
#from bokeh.sampledata.autompg import autompg
import pandas as pd

combPD = pd.read_csv('/processed_data/comb_ssaga-ephys/ssagadxNep3xmost_test.csv',
			na_values=['.'], converters = {'ID':str}, low_memory=False )

combB = combPD[['ID','sex','AGE_f0','d4dpdx_f0','delta_a-0']]

@bokeh_app.route("/bokeh/crossfilter/")
@object_page("crossfilter")
def make_crossfilter():
    #autompg['cyl'] = autompg['cyl'].astype(str)
    #autompg['origin'] = autompg['origin'].astype(str)
    app = CrossFilter.create(df=combB)
    return app