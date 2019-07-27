from rtlsdr import RtlSdr
import numpy as np

# rtlsdr config 
sdr = RtlSdr()
sdr.sample_rate = 1.92e6
sdr.gain = 'auto'
    #sdr.set_bandwidth = 8000
    #sdr.set_freq_correction = 30
center_freq = (28.8e6, 70e6)

# list devices for identification
devices = ['radio1', 'radio2']

# Global variables
all_train_samples = []
all_train_labels  = []
all_train_names   = []

all_test_samples = []
all_test_labels  = []
all_test_names   = []


for index in range(len(devices)):
    cf = center_freq[index]
    labels = []
    sample_names = []

    sdr.center_freq = cf

    #collect 2 million samples for each radio
    iq_samples = sdr.read_samples(1000000)

    real = np.real(iq_samples) 
    imag = np.imag(iq_samples)

    iq_seq = np.ravel(np.column_stack((real, imag)))

    # 50-50 split
    train_samples = iq_seq[:1000192]
    test_samples  = iq_seq[1000192:1999872]

    # Prepare training data
    train_samples = train_samples.reshape(3907, 2, 128)

    for sample in train_samples:
        #labels
        label  = np.zeros(len(devices))
        label[index] = 1.0
        labels.append(label)
        #names
        name = 'radio1'
        sample_names.append(name)

    train_samples = train_samples.reshape(-1, 2, 128, 1)
    train_labels = np.array(labels)
    train_sample_names = np.array(sample_names)

    all_train_samples.append(train_samples)
    all_train_labels.append(train_labels)
    all_train_names.append(train_sample_names)

    # Prepare test data

    test_samples = test_samples.reshape(3905, 2, 128)

    for sample in test_samples:
        #labels
        label  = np.zeros(len(devices))
        label[index] = 1.0
        labels.append(label)
        #names
        name = 'radio1'
        sample_names.append(name)

    test_samples = test_samples.reshape(-1, 2, 128, 1)
    test_labels = np.array(labels)
    test_sample_names = np.array(sample_names)

    all_test_samples.append(test_samples)
    all_test_labels.append(test_labels)
    all_test_names.append(test_sample_names)

# Concatenate samples from radios

train_data  = np.concatenate((all_train_samples[0], all_train_samples[1]))
train_label = np.concatenate((all_train_labels[0], all_train_labels[1]))
train_names = np.concatenate((all_train_names[0], all_train_names[1]))

test_data  = np.concatenate((all_test_samples[0], all_test_samples[1]))
test_label = np.concatenate((all_test_labels[0], all_test_labels[1]))
test_names = np.concatenate((all_test_names[0], all_test_names[1]))

# Save to file
 
np.save("train_samples.npy", train_data)
np.save("train_labels.npy", train_label)
np.save("train_sample_names.npy", train_names)

np.save("test_samples.npy", test_data)
np.save("test_labels.npy", test_label)
np.save("test_sample_names.npy", test_names)

sdr.close()