'''tools and classes to manage HBNL processing
'''
import sys, inspect
processing_module = sys.modules[__name__]
import pandas as pd

from .organization import MongoBacked, MongoConn, flatten_dict
# from .quest_import import quest_sparser_sub
# from .master_info import (subjects_sparser_sub, subjects_sparser_add,
#                             session_sadd)
def characterize( input ):
    ''' general function to describe input
    '''
    return { 'type': type(input),
            'size': sys.getsizeof(input)}

class ProcessBacked( MongoBacked ):

    Mdb = MongoConn['process']


    def store(s):  # overloading store method

        if 'store_id' in dir(s):
            s.update_query = {'_id':s.store_id}
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
        s.data_func = data_func

        s.data_for_db()

    def get_data(s):

        if s.data_func:
            return s.data_func(s.group)
        else:    
            return s.group

    def data_for_db(s):
        s.data = { 'group':str(s.group),
                    'data func':str(s.data_func) }


class pipelineStep( ProcessBacked ):
    ''' General representation of a processing step
    '''

    collection = 'steps'

    def_description = {'input type': None,
                        'process function': characterize,
                        'named inputs':{},
                        'output type': None}

    def __init__(s,description_in= {} ):

        s.description = s.def_description.copy()
        if not isinstance(description_in,dict):
            description = {'process function':description_in}
        else: description = description_in
        s.description.update(description)
        s.data_for_db()
        s.store()

    def data_for_db(s):

        s.data = s.description.copy()
        if 'store_name' in dir(s.data['process function']):
            s.data['process function'] = s.data['process function'].store_name
        else:  s.data['process function'] = 'unknown'


    def execute(s,input):

        return s.description['process function'](input,**s.description['named inputs'])


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
        # type checking here is fraught - using presence of execute method to identify pipelineStep
        #print(dir(processing_module))
        #print('test type in add_step ', isinstance(step, pipelineStep), type(step))
        #if not isinstance(step, pipelineStep):
        if 'execute' not in dir( step ):
            step = pipelineStep( step_in )

        if order == 'end':
            s.steps.append( step )

        s.data_for_db()

    def data_for_db(s):
        s.data = {'steps':[ st.store_id for st in s.steps ]}
        if 'store_name' in dir(s.init_fun):
            s.data['init function'] = s.init_fun.store_name


class workingPipe( ProcessBacked ):

    collection = 'work'

    def __init__(s,pipe,batch):

        s.pipe = pipe
        s.batch = batch

        data = s.pipe.load_batch(s.batch)
        s.step_io_data = [ data ]
        s.execution_status = len(s.pipe.steps)*[None]
        s.data_for_db()
        s.completed = False
        s.store()

    def execute_step(s,step_ind=None):
        ''' returns True if step runs, False if pipeline already completed
        '''

        if step_ind == None:
            step_ind = len(s.step_io_data)-1
        else:
            s.step_io_data = s.step_io_data[:step_ind]

        if step_ind < len(s.pipe.steps):

            s.step_io_data.append( s.pipe.steps[step_ind].execute( 
                                            s.step_io_data[ step_ind ] ) )
            s.execution_status[step_ind] = 'done'
            s.data_for_db()
            s.store()

            if s.execution_status[-1] == 'done':
                s.completed = True

            return True

        else:
            return False

    def assemble_step_inputs(s):
        s.step_inputs = []
        s.combined_inputs = {}
        for step in s.pipe.steps:
            inputs = s.pipe.description['named inputs']
            s.step_inputs.append( inputs )
            for k,v in inputs.items():
                if k in s.combined_inputs:
                    comb = s.combined_inputs[k]
                    if type(comb) == set:
                        s.combined_inputs[k] = comb.add(v)
                    else:
                        s.combined_inputs[k] = set([comb, v])
                else:
                    s.combined_inputs[k] = v

        for k,v in s.combined_inputs.items():
            if type(v) == set:
                if len(v) == 1:
                    s.combined_iputs[k] = v.pop()
        

    def data_for_db(s):
        if 'store_id' not in dir( s.batch ):
            s.batch.store()
        if 'store_id' not in dir( s.pipe ):
            s.pipe.store()
        s.data = {'batch': s.batch.store_id,
                  'pipe': s.pipe.store_id,
                  'step execution': s.execution_status
                  }


'''
Test pattern with simple string operations:

def test_fun(ins):
    return ins + ' processed'
test_fun.store_name = 'tester'

test_batch = PR.processingBatch(group='test',data_func=str.upper)
test_fun_ST = PR.pipelineStep({'process function':test_fun})
test_pipe = PR.pipe(steps=[test_fun,test_fun_ST]) # can pass raw function or pipelineStep object
test_Wpipe = PR.workingPipe( test_pipe, test_batch )

exres = test_Wpipe.execute_step()
# show data
test_Wpipe.step_io_data

'''