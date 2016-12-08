'''tools and classes to manage HBNL processing
'''
import sys
import pandas as pd

from .organization import MongoBacked, MongoConn, flatten_dict
from .quest_import import quest_sparser_sub
from .master_info import (subjects_sparser_sub, subjects_sparser_add,
                            session_sadd)
def characterize( input ):
    ''' general function to describe input
    '''
    return { 'type': type(input),
            'size': sys.getsizeof(input)}

class ProcessBacked( MongoBacked ):

    Mdb = MongoConn['process']


    def store(s):  # overloading store method

        if 'store_id' in dir(s):
            s.update()
        else:
            insert = s.store_track()
            s.store_id = insert.inserted_id
        #return insert

class processingBatch( ProcessBacked ):

    collection = 'batches'

    def __init__(s, group= None, data_func = None ):
        ''' group should have a collection name, and and u_id list 
        '''
        s.group = group
        s.dataFun = data_func

        s.data_for_db()

    def get_data(s):

        if s.dataFun:
            return s.dataFun(s.group)
        else:    
            return s.group

    def data_for_db(s):
        s.data = { 'group':str(s.group) }

class pipe( ProcessBacked ):

    collection = 'pipes'

    def __init__(s, steps=[], init_fun=None):
        s.init_fun = init_fun
        s.steps = []
        [ s.add_step(st) for st in steps ]

    def load_batch(s,batch):

        batchData = batch.get_data()

        if s.init_fun:
            data = s.init_fun(batchData)
        else: data = batchData

        return data

    def add_step(s, step_in, order='end'):
        ''' place step in pupe according to order (only end for now)
        '''

        step = step_in
        print('test type in add_step ', isinstance(step,pipelineStep), type(step))
        if not isinstance(step,pipelineStep):
            step = pipelineStep( step_in )
        
        if order == 'end':
            s.steps.append( step )

        s.data_for_db()

    def data_for_db(s):
        s.data = {'steps':[ st.store_id for st in s.steps ]}
        if 'store_name' in dir(s.init_fun):
            s.data['init function'] = s.init_fun.store_name


class pipelineStep( ProcessBacked ):
    ''' General representation of a processing step
    '''

    collection = 'steps'

    def_description = {'input type': None,
                        'process function': characterize,
                        'output type': None}

    def __init__(s,description= {} ):

        s.description = s.def_description.copy()
        s.description.update(description)
        s.data_for_db()

    def data_for_db(s):

        s.data = s.description.copy()
        if 'store_name' in dir(s.data['process function']):
            s.data['process function'] = s.data['process function'].store_name
        else:  s.data['process function'] = 'unknown'


    def execute(s,input):

        return s.description['process function'](input)


class workingPipe( ProcessBacked ):

    collection = 'work'

    def __init__(s,pipe,batch):

        s.pipe = pipe
        s.batch = batch

        data = s.pipe.load_batch(s.batch)
        s.step_io_data = [ data ]
        s.execution_status = len(s.pipe.steps)*[None]

    def execute_step(s,step_ind=None):

        if step_ind == None:
            step_ind = len(s.step_io_data)-1
        else:
            s.step_io_data = s.step_io_data[:step_ind]

        s.step_io_data.append( s.pipe.steps[step_ind].execute( 
                                        s.step_io_data[ step_ind ] ) )
        s.execution_status[step_ind] = 'done'

        s.data_for_db()

    def data_for_db(s):
        if 'store_id' not in dir( s.batch ):
            s.batch.store()
        if 'store_id' not in dir( s.pipe ):
            s.pipe.store()
        s.data = {'batch': s.batch.store_id,
                  'pipe': s.pipe.store_id,
                  'step execution': s.execution_status
                  }
