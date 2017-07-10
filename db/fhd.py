''' compiling FHD '''

import numpy as np

from . import database as D
from . import compilation as C
from .demog_tools import default_field_eponyms, calc_ph, calc_fhd, calc_fhd_catnorm, calc_fhd_pathlen
from .utils.compilation import collapse_POP, calc_PH

correct_ped_genetically = False


def calc_coga_ph(row, aff_col='cor_alc_dep_dx', pop_col='POP'):
    ''' row apply function for calculating the POP + PH variable '''

    parent_col = 'parent_' + aff_col
    if row[pop_col] == 'COGA' and row[parent_col] == 1:
        return 1
    elif row[pop_col] == 'CTL' and row[parent_col] == 0:
        return 0
    return np.nan


def build_fhd_df():
    # subjects dataframe

    core_cols = ['ID',
                 'alc_dep_dx',
                 'ald5dx',
                 'ald5sx_max_cnt',
                 ]
    core_proj = {col: 1 for col in core_cols}
    core_proj['_id'] = 0

    docs = D.Mdb['core'].find({}, core_proj)
    core_ID_df = C.buildframe_fromdocs(docs, inds=['ID'])
    renamer = {col: 'cor_' + col for col in core_ID_df.columns}
    core_ID_df.rename(columns=renamer, inplace=True)

    rels_cols = default_field_eponyms.copy()
    rels_proj = {col:1 for col in rels_cols}
    rels_proj.update({'_id': 0})

    core_ID_df = C.join_collection(core_ID_df, 'allrels', add_proj=rels_proj, prefix='')

    sub_cols = ['POP']
    sub_proj = {col:1 for col in sub_cols}
    core_ID_df_pop = C.join_collection(core_ID_df, 'subjects', add_proj=sub_proj, prefix='')

    sub_df = core_ID_df_pop.ix[core_ID_df_pop['POP'].isin(['COGA', 'COGA-Ctl', 'IRPG', 'IRPG-Ctl'])]
    sub_df['POP'] = sub_df['POP'].apply(collapse_POP)

    # all rels fam df

    fam_df = C.get_famdf(sub_df)
    fam_df = C.join_collection(fam_df, 'core', add_proj=core_proj,)
    fam_df['cor_ald5sx_max_cnt_log'] = np.log1p(fam_df['cor_ald5sx_max_cnt'])

    # calculating FHD

    sub_df_fhd = sub_df.copy()

    # calculating FHD based on dichotomous affectedness

    dich_aff_cols = ['cor_alc_dep_dx', 'cor_ald5dx']
    dich_fh_funcs = [calc_ph, calc_fhd, calc_fhd_pathlen]

    for ac in dich_aff_cols:
        for ff in dich_fh_funcs:
            fh_df = ff(sub_df, fam_df, aff_col=ac)
            sub_df_fhd = sub_df_fhd.join(fh_df)

    sub_df_fhd['parentCOGA_cor_alc_dep_dx'] = sub_df_fhd.apply(calc_coga_ph, axis=1, args=['cor_alc_dep_dx'])
    sub_df_fhd['parentCOGA_cor_ald5dx'] = sub_df_fhd.apply(calc_coga_ph, axis=1, args=['cor_ald5dx'])

    # calculating FHD based on symptom counts (log-transformed, here)

    count_aff_cols_maxes = {'cor_ald5sx_max_cnt_log': np.log1p(11)}
    count_fh_funcs = [calc_fhd, calc_fhd_pathlen]

    for ac, ac_max in count_aff_cols_maxes.items():
        for ff in count_fh_funcs:
            fh_df = ff(sub_df, fam_df, aff_col=ac)
            tmp_norm_cols = [col for col in fh_df.columns if 'fhd' in col]
            for col in tmp_norm_cols:
                fh_df[col] = fh_df[col] / ac_max
            sub_df_fhd = sub_df_fhd.join(fh_df)

    # retain only estimates based on 4+ relatives

    nrels_cols = [col for col in sub_df_fhd.columns if col.startswith('nrels')]

    sub_df_fhd_relthresh = sub_df_fhd.copy()

    for nc in nrels_cols:
        col_suffix = nc[5:]
        less4_bool = sub_df_fhd_relthresh[nc] < 4
        fhd_cols = ['fhdsum'+col_suffix, 'fhdratio'+col_suffix]
        sub_df_fhd_relthresh.ix[less4_bool, fhd_cols] = np.nan

    # drop irrelevant columns

    irrelevant_columns = ['POP', 'cor_alc_dep_dx', 'cor_ald5dx', 'cor_ald5sx_max_cnt',]
    sub_df_fhd_relthresh.drop(irrelevant_columns , axis=1, inplace=True)

    return sub_df_fhd_relthresh
