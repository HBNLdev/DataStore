'''Tests for hbnl database
'''

import compilation as C
import organization as O

def test_check_collinputs():

    # Fails with nonexistent collection
    assert C.check_collinputs('wrong') == False

    # Passes