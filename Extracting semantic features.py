import gensim
import matplotlib as mpl
from utils import *
from gensim.models import word2vec
for module in mpl,np,gensim:
    print(module.__name__,module.__version__)
def word_vector_list(index2word):
    word_vec = []
    for i in index2word:
        word_vec.append(word2vec.wv[str(i)])
    return word_vec
def Sequence_to_word2vec(sequence):
    zero_padding = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # print(zero_padding)
    return_list = []
    # words_list = ["CNF"]
    for i in range(len(sequence)):
        Single_temporary_sequence = []
        for j in range(0, len_seq):
            if len(str(sequence[i][j: j + 1]).replace("X", "")) == 0:
                Single_temporary_sequence.append(zero_padding)
            else:
                Single_temporary_sequence.append(word2vec.wv[str(sequence[i][j: j + 1]).replace("X", "").strip()])
        return_list.append(Single_temporary_sequence)
    return return_list

print("Get a corpus......")
train_Pro_Id_198, Init_train_seqs_198, train_slice_seq_198, train_seq_labels_198 = read_file_splic(
    path=r"./data/",
    file_name="RB198.txt",
    len_seq=21, head_notes=3)

# *************   Parameter configuration  ************#
fea_dim = 25
window_size = 5
print("training  word2vec......")
# ****************************   training  word2vec   ****************************#
word2vec = word2vec.Word2Vec(sentences=Init_train_seqs_198, size=fea_dim, window=window_size,
                             sg=0, hs=0, negative=5, min_count=5, seed=18, iter=200, workers=1)
index2word = word2vec.wv.index2word
#print(index2word)
word_vector_200 = word_vector_list(index2word=index2word)

# # ************************ save  word vec  200  ******************************************************#
# np.savez(file=r"F:\program\jupyter\WordtoXGBPRBing\Data processing\word vec\\word_vector_200.npz",
#          index2word=index2word,
#          word_vector_200=word_vector_200)

# **** Calculate the cosine similarity of word vector features  200 ***************#
print("Calculate the cosine similarity of word vector features")
calc_cosin_simi(word_vec=word_vector_200,
                save_path=r"./figure/cosine similarity/",
                file_name=r"word2vec similarity.pdf")

## Sequence division and feature extraction

len_seq = 39  # Dividing sequence length
file_name = str(len_seq)
save_path = r'./Semantic features/'

print("Sequence division......")
#Training data
train_Pro_Id, Init_train_seqs, train_slice_seq, train_seq_labels = read_file_splic(
    path=r"./data/",
    file_name="RB198.txt",
    len_seq=len_seq, head_notes=3)

###Test Data
test_Pro_Id, Init_test_seqs, test_slice_seq, test_seq_labels = read_file_splic(
    path=r"./data/",
    file_name="RB111.txt", len_seq=len_seq, head_notes=3)
print("Extraction Semantic features...... ")
# ********** The delineated training data sequences are converted into word vectors  ****************#
train_word2vec_data = Sequence_to_word2vec(train_slice_seq)

# ********** The delineated test data sequences are converted into word vectors  ****************#
test_word2vec_data = Sequence_to_word2vec(test_slice_seq)

# *********************  Converting to numpy format data  ***********************************#
train_data = np.array(train_word2vec_data).reshape(-1, len_seq, 25)
# train_label = np.array(train_seq_labels)
train_label = np.array([int(float(i)) for i in train_seq_labels])
train_label = np.array(train_label)

# *********************  Converting to numpy format data  ***********************************#
test_data = np.array(test_word2vec_data).reshape(-1, len_seq, 25)
# test_label =np.array(test_seq_labels)
test_label = [np.array(int(float(i))) for i in test_seq_labels]
test_label = np.array(test_label)

# *********************  Save training and test data  ***********************************#
print("Save semantic feature extraction of training set and test set...... ")
np.savez(file=save_path + "Train_Test_numpy_data", train_data=train_data,
         train_label=train_label, test_data=test_data, test_label=test_label)
print("train_data.shape:",train_data.shape)
print("test_data.shape:",test_data.shape)
print("Done")
