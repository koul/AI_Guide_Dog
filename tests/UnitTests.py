from labeler import Labeler, Selector
test_file = "../data/TCSSimpleWalk/basic_test.csv"

def test_labeler():
    labeler_ = Labeler.Labeler(test_file, lookforward=90)
    modified_df = labeler_.add_direction()
    print('done')

test_labeler()



