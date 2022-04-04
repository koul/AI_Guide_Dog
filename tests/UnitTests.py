from labeler import Labeler, Selector
test_file = "../data/TCSSimpleWalk/basic_test.csv"

def test_labeler():
    labeler_ = Labeler.Labeler(test_file)
    modified_df = labeler_.add_direction()
    print('done')

test_labeler()



