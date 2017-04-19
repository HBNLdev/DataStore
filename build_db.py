from datetime import datetime

import db.collection as collection_module

db_confirmed = False

while not db_confirmed:

    db_prompt = 'note that you are using the DB named {}, continue? y/n: '.format(collection_module.Mdb.name)
    go_ahead = input(db_prompt)

    if go_ahead == 'y':
        db_confirmed = True
    else:
        rename_db_prompt = 'enter the name of the DB you would like to use: '
        new_db_name = input(rename_db_prompt)
        collection_module.set_db(new_db_name)

for class_name in collection_module.build_order:

    class_instance = getattr(collection_module, class_name)()
    n_docs = class_instance.count()

    if n_docs > 0:

        if class_name in collection_module.incremental_collections:

            prompt = '{} contains {} docs but is incremental, re-run the build? y/n: '. \
                format(class_instance.collection_name, n_docs)
            do_rebuild = input(prompt)

            if do_rebuild != 'y':
                print('skipping', class_instance.collection_name)
                continue

        else:

            prompt = '{} already contains {} docs, do you want to rebuild? y/n: '. \
                format(class_instance.collection_name, n_docs)
            do_rebuild = input(prompt)

            if do_rebuild == 'y':
                class_instance.drop()
            else:
                print('skipping', class_instance.collection_name)
                continue

    print('building', class_instance.collection_name)

    t0 = datetime.now()
    class_instance.build()
    t1 = datetime.now()

    print('build of {} completed, taking {}'.format(class_instance.collection_name, t1 - t0))
