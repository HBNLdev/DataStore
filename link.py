'''link collections'''

import organization as O

def create_links(parent_coll_name,link_coll_name,link_fields,add_query={}):
    '''store list of _id's in link_coll_name collection
       under link_coll_name fileld of '_link' document in
       each appropriate parent_coll document

       link_fields can be a string or list/tuple for multiple fields
    '''

    if type(link_fields) == str:
        link_fields = [ link_fields ]
    print(type(link_fields),link_fields)

    parent_coll = O.Mdb[parent_coll_name]
    link_coll = O.Mdb[link_coll_name]
    check_link_fields_query = \
            { lf:{'$exists':True} for lf in link_fields \
                    if lf not in add_query }
    add_query.update(check_link_fields_query)
    parent_docs = parent_coll.find(add_query)
    #if debug:
    lcount = ldocs = 0
    print(parent_docs.count(), 'matching documents in',parent_coll)
    for pd in parent_docs:
        link_query = {'$and':[ {lf:pd[lf]} for lf in link_fields] }
        link_docs = link_coll.find(link_query)
        lcount += link_docs.count()
        if link_docs.count():#link_docs.count():
            link_ids = [ d['_id'] for d in link_docs ]
            parent_coll.update({'_id':pd['_id']},{ '$set':{'_links.'+link_coll_name:link_ids} })
            #if debug:
            ldocs+=1

    print(ldocs, 'updated with',lcount, 'total links')

