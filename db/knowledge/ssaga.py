''' knowledge about the SSAGA and the core phenotype file '''

ssaga_cols = [
    'AL1',
    'AL1AgeOns',
    'AL4a',
    'AL4e1',
    'AL4e2',
    'AL4e3',
    'AL4fx',
    'AL4f1',
    'AL4f2',
    'AL4f3',
    'AL5',
    'AL6',
    'AL6a',
    'AL8',
    'AL8a',
    'AL8AgeOns',
    'AL8d',
    'AL9',
    'AL9bAgeOns',
    'AL9bAgeRec',
    'AL16b',
    'AL16AgeOns',
    'AL17b',
    'AL17AgeOns',
    'AL17AgeRec',
    'AL19',
    'AL22AgeOns',
    'AL22AgeRec',
    'AL25AgeOns',
    'AL25AgeRec',
    'AL31AgeOns',
    'AL33AgeOns',
    'AL33AgeRec',
    'AL34AgeOns',
    'AL35AgeOns',
    'AL37aAgeOns',
    'AL37aAgeRec',
    'AL37eAgeOns',
    'AL37eAgeRec',
    'AL43A',
    'AL45bAgeOns',
    'AL45bAgeRec', ]

ssaga_drink_AL3_cols = ['AL3WEEK_BEER1',
                        'AL3WEEK_BEER2',
                        'AL3WEEK_BEER3',
                        'AL3WEEK_BEER4',
                        'AL3WEEK_BEER5',
                        'AL3WEEK_BEER6',
                        'AL3WEEK_BEER7',
                        'AL3WEEK_LIQUOR1',
                        'AL3WEEK_LIQUOR2',
                        'AL3WEEK_LIQUOR3',
                        'AL3WEEK_LIQUOR4',
                        'AL3WEEK_LIQUOR5',
                        'AL3WEEK_LIQUOR6',
                        'AL3WEEK_LIQUOR7',
                        'AL3WEEK_OTHER1',
                        'AL3WEEK_OTHER2',
                        'AL3WEEK_OTHER3',
                        'AL3WEEK_OTHER4',
                        'AL3WEEK_OTHER5',
                        'AL3WEEK_OTHER6',
                        'AL3WEEK_OTHER7',
                        'AL3WEEK_WINE1',
                        'AL3WEEK_WINE2',
                        'AL3WEEK_WINE3',
                        'AL3WEEK_WINE4',
                        'AL3WEEK_WINE5',
                        'AL3WEEK_WINE6',
                        'AL3WEEK_WINE7', ]

ssaga_drink_AL4_cols = ['AL4WEEK_BEER1',
                        'AL4WEEK_BEER2',
                        'AL4WEEK_BEER3',
                        'AL4WEEK_BEER4',
                        'AL4WEEK_BEER5',
                        'AL4WEEK_BEER6',
                        'AL4WEEK_BEER7',
                        'AL4WEEK_LIQUOR1',
                        'AL4WEEK_LIQUOR2',
                        'AL4WEEK_LIQUOR3',
                        'AL4WEEK_LIQUOR4',
                        'AL4WEEK_LIQUOR5',
                        'AL4WEEK_LIQUOR6',
                        'AL4WEEK_LIQUOR7',
                        'AL4WEEK_OTHER1',
                        'AL4WEEK_OTHER2',
                        'AL4WEEK_OTHER3',
                        'AL4WEEK_OTHER4',
                        'AL4WEEK_OTHER5',
                        'AL4WEEK_OTHER6',
                        'AL4WEEK_OTHER7',
                        'AL4WEEK_WINE1',
                        'AL4WEEK_WINE2',
                        'AL4WEEK_WINE3',
                        'AL4WEEK_WINE4',
                        'AL4WEEK_WINE5',
                        'AL4WEEK_WINE6',
                        'AL4WEEK_WINE7', ]

dx_ssaga_cols = ['ALD4DPSX',
                 'ALD4DPDX',
                 'ALD4DPAO',
                 'ALD4ABDX',
                 'ALD4ABSX',
                 'ALD4D1',
                 'ALD4D2',
                 'ALD4D3',
                 'ALD4D4',
                 'ALD4D5',
                 'ALD4D6',
                 'ALD4D7',
                 'ALD4ABA1',
                 'ALD4ABA2',
                 'ALD4ABA3',
                 'ALD4ABA4',
                 'TBD4DPSX',
                 'TBD4DPDX',
                 'TBD4DPAO',
                 'MJD4DPSX',
                 'MJD4DPDX',
                 'MJD4DPAO',
                 'MJD4ABDX',
                 'MJD4ABSX',
                 'COD4DPDX',
                 'COD4DPSX',
                 'COD4DPAO',
                 'COD4ABDX',
                 'COD4ABSX',
                 'STD4DPDX',
                 'STD4DPSX',
                 'STD4DPAO',
                 'STD4ABDX',
                 'STD4ABSX',
                 'SDD4DPDX',
                 'SDD4DPSX',
                 'SDD4DPAO',
                 'SDD4ABDX',
                 'SDD4ABSX',
                 'OPD4DPDX',
                 'OPD4DPSX',
                 'OPD4DPAO',
                 'OPD4ABDX',
                 'OPD4ABSX',
                 'OTHERDRUGD4DPDX',
                 'OTHERDRUGD4DPSX',
                 'OTHERDRUGD4DPAO',
                 'OTHERDRUGD4ABDX',
                 'OTHERDRUGD4ABSX',
                 'DPD4DX',
                 'DPD4SX',
                 'ASD4DX',
                 'ASD4A',
                 'ASD4C',
                 'ASD4ASX_CLEAN',
                 'ASD4ASX_CLEANORDIRTY',
                 'ASD4CSX_CLEAN',
                 'ASD4CSX_CLEANORDIRTY',
                 'ADD4DX',
                 'ADD4A1SX',
                 'ADD4A2HYPSX',
                 'ADD4A2IMPSX',
                 'ADD4A2SX',
                 'ODD4DX',
                 'ODD4SX',
                 'ODD4BCOUNT',
                 'PTD4DX',
                 'PTD4CRITBSX',
                 'PTD4CRITCSX',
                 'PTD4CRITDSX',
                 'OCOBD4DX',
                 'OCCPD4DX',
                 'SPD4DX',
                 'PND4DX',
                 'PND4NUMSX',
                 'AGD4DX', ]

dx_ssaga_ald_cols = ['ALD4DPSX',
                 'ALD4DPDX',
                 'ALD4DPAO',
                 'ALD4ABDX',
                 'ALD4ABSX',
                 'ALD4D1',
                 'ALD4D2',
                 'ALD4D3',
                 'ALD4D4',
                 'ALD4D5',
                 'ALD4D6',
                 'ALD4D7',
                 'ALD4ABA1',
                 'ALD4ABA2',
                 'ALD4ABA3',
                 'ALD4ABA4',]

core_cols = ['alc_dep_dx',
             'alc_dep_ons',
             'alc_dep_max_sx1',
             'alc_dep_max_sx2',
             'alc_dep_max_sx3',
             'alc_dep_max_sx4',
             'alc_dep_max_sx5',
             'alc_dep_max_sx6',
             'alc_dep_max_sx7',
             'alc_dep_max_sx_cnt',
             'aldp_maxsx_whint',
             'ever_drink',
             'alc_abuse',
             'alc_abuse_ons',
             'alc_abuse_max_sx1',
             'alc_abuse_max_sx2',
             'alc_abuse_max_sx3',
             'alc_abuse_max_sx4',
             'alc_abuse_max_sx_cnt',
             'max_drinks',
             'max_dpw',
             'max_dpw_pwk',
             'age_first_drink',
             'age_last_drink',
             'regular_drinking',
             'reg_drink_ons',
             'ever_got_drunk',
             'age_first_got_drunk',
             'ald5dx',
             'ald5sx_cnt',
             'ald5sx_max_cnt',
             'ald5dx_sev',
             'ald5dx_max_sev',
             'ald5_first_whint',
             'ald5_max_whint',
             'tb_dep_dx',
             'tb_dep_sx_cnt',
             'tb_dep_ons',
             'ftnd_4',
             'max_cpd',
             'age_first_cig',
             'age_first_reg_cig',
             'mj_dep_dx',
             'mj_dep_ons',
             'mj_dep_sx_cnt',
             'ever_mj',
             'age_first_use_mj',
             'mj_dep_max_sx_cnt',
             'mj_abuse',
             'mj_abuse_sx_cnt',
             'mj_abuse_max_sx_cnt',
             'mjd5dx',
             'mjd5sx_cnt',
             'mjd5sx_max_cnt',
             'mjd5dx_sev',
             'mjd5dx_max_sev',
             'co_dep_dx',
             'co_dep_ons',
             'co_dep_sx_cnt',
             'ever_co',
             'age_first_use_co',
             'co_dep_max_sx_cnt',
             'co_abuse',
             'co_abuse_sx_cnt',
             'co_abuse_max_sx_cnt',
             'cod5dx',
             'cod5sx_cnt',
             'cod5sx_max_cnt',
             'cod5dx_sev',
             'cod5dx_max_sev',
             'st_dep_dx',
             'st_dep_ons',
             'st_dep_sx_cnt',
             'ever_st',
             'age_first_use_st',
             'st_dep_max_sx_cnt',
             'st_abuse',
             'st_abuse_sx_cnt',
             'st_abuse_max_sx_cnt',
             'std5dx',
             'std5sx_cnt',
             'std5sx_max_cnt',
             'std5dx_sev',
             'std5dx_max_sev',
             'sd_dep_dx',
             'sd_dep_ons',
             'sd_dep_sx_cnt',
             'ever_sd',
             'age_first_use_sd',
             'sd_dep_max_sx_cnt',
             'sd_abuse',
             'sd_abuse_sx_cnt',
             'sd_abuse_max_sx_cnt',
             'age_first_use_op',
             'sdd5dx',
             'sdd5sx_cnt',
             'sdd5sx_max_cnt',
             'sdd5dx_sev',
             'sdd5dx_max_sev',
             'op_dep_dx',
             'op_dep_ons',
             'op_dep_sx_cnt',
             'ever_op',
             'age_last_use_op',
             'op_dep_max_sx_cnt',
             'op_abuse',
             'op_abuse_sx_cnt',
             'op_abuse_max_sx_cnt',
             'opd5dx',
             'opd5sx_cnt',
             'opd5sx_max_cnt',
             'opd5dx_sev',
             'opd5dx_max_sev', ]


core_ald_cols = ['alc_dep_dx',
             'alc_dep_ons',
             'alc_dep_max_sx1',
             'alc_dep_max_sx2',
             'alc_dep_max_sx3',
             'alc_dep_max_sx4',
             'alc_dep_max_sx5',
             'alc_dep_max_sx6',
             'alc_dep_max_sx7',
             'alc_abuse_max_sx1',
             'alc_abuse_max_sx2',
             'alc_abuse_max_sx3',
             'alc_abuse_max_sx4',
             'alc_dep_max_sx_cnt',
             'aldp_first_whint',
             'aldp_maxsx_whint',
             'ever_drink',
             'alc_abuse',
             'alc_abuse_ons',
             'alc_abuse_max_sx_cnt',
             'max_drinks',
             'max_dpw',
             'max_dpw_pwk',
             'age_first_drink',
             'age_last_drink',
             'regular_drinking',
             'reg_drink_ons',
             'ever_got_drunk',
             'age_first_got_drunk',
             'ald5dx',
             'ald5sx_cnt',
             'ald5sx_max_cnt',
             'ald5dx_sev',
             'ald5dx_max_sev',
             'ald5_first_whint',
             'ald5_max_whint',
             'age_last_intvw',
             ]

core_dx_cols = ['alc_dep_dx',
                'alc_dep_dx_sx_cnt',
                'alc_dep_max_sx_cnt',
                'alc_abuse',
                'alc_abuse_sx_cnt',
                'alc_abuse_max_sx_cnt',
                'ald5dx',
                'ald5sx_cnt',
                'ald5sx_max_cnt',
                'ald5dx_sev',
                'ald5dx_max_sev',
                'tb_dep_dx',
                'tb_dep_sx_cnt',
                'ftnd_4',
                'mj_dep_dx',
                'mj_dep_sx_cnt',
                'mj_dep_max_sx_cnt',
                'mj_abuse',
                'mj_abuse_sx_cnt',
                'mj_abuse_max_sx_cnt',
                'mjd5dx',
                'mjd5sx_cnt',
                'mjd5sx_max_cnt',
                'mjd5dx_sev',
                'mjd5dx_max_sev',
                'co_dep_dx',
                'co_dep_sx_cnt',
                'co_dep_max_sx_cnt',
                'co_abuse',
                'co_abuse_sx_cnt',
                'co_abuse_max_sx_cnt',
                'cod5dx',
                'cod5sx_cnt',
                'cod5sx_max_cnt',
                'cod5dx_sev',
                'cod5dx_max_sev',
                'st_dep_dx',
                'st_dep_sx_cnt',
                'st_dep_max_sx_cnt',
                'st_abuse',
                'st_abuse_sx_cnt',
                'st_abuse_max_sx_cnt',
                'std5dx',
                'std5sx_cnt',
                'std5sx_max_cnt',
                'std5dx_sev',
                'std5dx_max_sev',
                'sd_dep_dx',
                'sd_dep_sx_cnt',
                'sd_dep_max_sx_cnt',
                'sd_abuse',
                'sd_abuse_sx_cnt',
                'sd_abuse_max_sx_cnt',
                'sdd5dx',
                'sdd5sx_cnt',
                'sdd5sx_max_cnt',
                'sdd5dx_sev',
                'sdd5dx_max_sev',
                'op_dep_dx',
                'op_dep_sx_cnt',
                'op_dep_max_sx_cnt',
                'op_abuse',
                'op_abuse_sx_cnt',
                'op_abuse_max_sx_cnt',
                'opd5dx',
                'opd5sx_cnt',
                'opd5sx_max_cnt',
                'opd5dx_sev',
                'opd5dx_max_sev', ]

core_aud_dx_cols = ['alc_dep_dx',
                    'alc_dep_ons',
                    'alc_dep_dx_sx_cnt',
                    'alc_dep_max_sx_cnt',
                    'alc_abuse',
                    'alc_abuse_sx_cnt',
                    'alc_abuse_max_sx_cnt',
                    'ald5dx',
                    'ald5sx_cnt',
                    'ald5sx_max_cnt',
                    'ald5dx_sev',
                    'ald5dx_max_sev',
                    'age_last_intvw',
                    'aldp_maxsx_whint',
                    'aldp_first_whint',
                    'ald5_max_whint',
                    'ald5_first_whint',
                    ]

core_dsm5_sx_cols = ['alc_dep_max_sx1',
                     'alc_dep_max_sx2',
                     'alc_dep_max_sx3',
                     'alc_dep_max_sx4',
                     'alc_dep_max_sx5',
                     'alc_dep_max_sx6',
                     'alc_dep_max_sx7',
                     'alc_abuse_max_sx1',
                     'alc_abuse_max_sx2',
                     'alc_abuse_max_sx4', ]
