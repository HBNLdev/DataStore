import master_info as mi
import organization as O


def subjects():

    mi_mtime = mi.load_master()
    source_rec = O.Mdb['subjects'].find({'_source': {'$exists': True}})

    # compare source file names and date modified

    if mi.path == source_rec['_s'][0] and mi_mtime <= source_rec['_s'][1]:
        return  # same path and older/same mdate, no update required

    else:  # new masterfile, do update

        old_ids = {rec['ID'] for rec in O.Mdb['subjects'].find(
            {'ID': {'$exists': True}})}
        new_ids = {mi.master['ID'].tolist()}  # sets
        add_ids = new_ids - old_ids

        addID_df = mi.master[mi.master['ID'].isin(add_ids)]
        for rec in addID_df.to_dict(orient='records'):
            sO = O.Subject(rec)
            sO.storeNaTsafe()
            # can do sessions here too

        sourceO = O.Source('subjects', [mi.path, mi_mtime])
        sourceO.update()


def sessions():

    mi_mtime = mi.load_master()
    source_rec = O.Mdb['subjects'].find({'_source': {'$exists': True}})

    # compare source file names and date modified

    if mi.path == source_rec['_s'][0] and mi_mtime <= source_rec['_s'][1]:
        return  # same path and older/same mdate, no update required

    else:  # new masterfile, do update

        old_uids = {(r['ID'], r['session']) for r in O.Mdb['subjects'].find(
            {'ID': {'$exists': True}})}
        new_uids = {(id, session) for (id, session) in zip(
            mi.master['ID'].tolist(), mi.master['session'].tolist())}  # sets
        add_uids = new_uids - old_uids

        addID_df = mi.master[mi.master['ID'].isin(add_uids)]
        for rec in addID_df.to_dict(orient='records'):
            sO = O.Subject(rec)
            sO.storeNaTsafe()

        sourceO = O.Source('subjects', [mi.path, mi_mtime])
        sourceO.update()


def erp():
    pass


def neuropsych_xml():
    pass
