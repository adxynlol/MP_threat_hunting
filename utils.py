from scapy.all import *
from scapy.layers import *
from scapy.layers.http import *
from scapy.layers.ssh import *
from scapy.contrib.mqtt import *
from scapy import *

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import random
import time
import builtins

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, BatchNormalization, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)
np.set_printoptions(suppress=True)

def plot_history(history):
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    
    num_epochs = len(history.epoch)
    epochs = [x+1 for x in history.epoch]
    
    ax1.plot(epochs, history.history["loss"], marker='.', label="train_loss")
    ax1.plot(epochs, history.history["val_loss"], marker='.', label="val_loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train and Validation Loss Over Epochs", fontsize=14)
    ax1.set_xticks(epochs[0::int(num_epochs/5)], epochs[0::int(num_epochs/5)])
    ax1.set_xlabel("Epochs")
    ax1.legend()
    ax1.grid()
    
    ax2.plot(epochs, history.history["accuracy"], marker='.', label="train_accuracy")
    ax2.plot(epochs, history.history["val_accuracy"], marker='.', label="val_accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Train and Validation Accuracy Over Epochs", fontsize=14)
    ax2.set_xticks(epochs[0::int(num_epochs/5)], epochs[0::int(num_epochs/5)])
    ax2.set_xlabel("Epochs")
    ax2.legend()
    ax2.grid()
    
    plt.show()
    
    return

def create_flows(pcap_file, attackers, victims):
    flows = {}
    for pkt in rdpcap(pcap_file):
        if pkt.haslayer(Ether) and pkt.haslayer(IP):
            src_mac = pkt[Ether].src
            dst_mac = pkt[Ether].dst

            src_ip = pkt[IP].src
            src_port = None
            dst_ip = pkt[IP].dst
            dst_port = None
            
            if pkt.haslayer(TCP):
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif pkt.haslayer(UDP):
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport

            if src_mac and src_ip and src_port and dst_mac and dst_ip and dst_port:

                if dst_mac in victims and src_mac in attackers:
                    flow_key = (src_mac, dst_mac, src_ip, dst_ip)
                elif src_mac in victims and dst_mac in attackers:
                    flow_key = (dst_mac, src_mac, dst_ip, src_ip)
                else:
                    continue
                
                if pkt.haslayer(TCP) and pkt.haslayer(Raw):
                    pkt[TCP].decode_payload_as(HTTP)
                
                if pkt.haslayer(HTTP):
                    if flow_key not in flows:
                        flows[flow_key] = []
                    flows[flow_key].append(pkt)

    return flows

def get_samples(flows):

    flow_data = []
    
    for _, packets in flows.items():
        
        flag = False
        
        samples = []
        times = []
        
        for pkt in packets:
            
            if pkt.haslayer(HTTP):

                if not flag:
                    t = float(pkt.time)
                    flag = True
                
                ip_packet = pkt[IP]

                ip_header = ip_packet.copy()
                ip_header.remove_payload()
                
                sample = list(bytes(ip_header))[:10]
    
                tcp_packet = ip_packet.payload
    
                tcp_header = tcp_packet.copy()
                tcp_header.remove_payload()
    
                s = list(bytes(tcp_header))
    
                sample = sample + s[:16] + s[18:] # remove just the checksum
                
                http_packet = tcp_packet.payload
    
                sample = sample + list(bytes(http_packet))
                
                sample = [x / 255.0 for x in sample]
                
                samples.append(sample)

                times.append(float(pkt.time)-t)
                
        flow_data.append([samples, times])
        
    return flow_data

def trim_or_pad(flows, fixed_length, max_num_packets):
    
    flow_data = get_samples(flows)
    
    datalist = []
    
    for flow in flow_data:
        
        samples = []

        count = 0
        
        for sample in flow[0]:

            count = count + 1
            
            if count > max_num_packets:
                break
            
            if len(sample) < fixed_length:
                s = sample + [0] * (fixed_length - len(sample))
            else:
                s = sample[:fixed_length]
            
            samples.append(s)
    
        datalist.append([samples, flow[1][:max_num_packets]])
        
    return datalist

def create_masks(flows, max_len):
    
    packets = []
    timestamps = []
    
    for flow in flows:
        packets.append(flow[0])
        timestamps.append(flow[1])
    
    padded_packets = tf.keras.utils.pad_sequences(
        packets,
        maxlen=max_len,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )

    padded_timestamps = tf.keras.utils.pad_sequences(
        timestamps,
        maxlen=max_len,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )
    
    attention_mask = tf.cast(padded_packets[..., 0] != 0, dtype=tf.float32)

    return padded_packets, padded_timestamps, attention_mask.numpy()

def create_subflows(d, t, m, max_num_packets, k=1):

    length = int(np.sum(m)/k)
    
    d_res = np.empty((length, d.shape[0], d.shape[1]))
    t_res = np.empty((length, t.shape[0]))
    m_res = np.empty((length, m.shape[0]))

    index = 0
    
    for i in range(1, length + 1, k):
        padded_packets = tf.keras.utils.pad_sequences(
            [d[:i]],
            maxlen=max_num_packets,
            dtype='float32',
            padding='post',
            truncating='post',
            value=0.0
        )
        d_res[index] = padded_packets

        padded_timestamps = tf.keras.utils.pad_sequences(
            [t[:i]],
            maxlen=max_num_packets,
            dtype='float32',
            padding='post',
            truncating='post',
            value=0.0
        )
        t_res[index] = padded_timestamps
        
        attention_mask = tf.cast(padded_packets[..., 0] != 0, dtype=tf.float32)
        m_res[index] = attention_mask

        index = index + 1
    
    return d_res, t_res, m_res

def remove_packet(flow, t, m, max_num_packets, length):
    """Randomly removes a packet from the flow."""
    index = random.randint(1, length - 1)

    padded_packets = tf.keras.utils.pad_sequences(
        [np.delete(flow, index, axis=0)],
        maxlen=max_num_packets,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )
    padded_packets = tf.squeeze(padded_packets)

    padded_timestamps = tf.keras.utils.pad_sequences(
        [np.delete(t, index, axis=0)],
        maxlen=max_num_packets,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )
    padded_timestamps = tf.squeeze(padded_timestamps)
    
    attention_mask = tf.cast(padded_packets[..., 0] != 0, dtype=tf.float32)
    
    length = int(np.sum(attention_mask))
    
    return padded_packets, padded_timestamps, attention_mask, length

def duplicate_packet(flow, t, m, max_num_packets, length):
    """Randomly duplicates a packet in the flow."""
    index = random.randint(1, length - 1)
    
    packet = flow[index]
    padded_packets = tf.keras.utils.pad_sequences(
        [np.insert(flow, index, packet, axis=0)],
        maxlen=max_num_packets,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )
    padded_packets = tf.squeeze(padded_packets)

    timestamp = (t[index] + t[index-1])/2
    padded_timestamps = tf.keras.utils.pad_sequences(
        [np.insert(t, index, timestamp, axis=0)],
        maxlen=max_num_packets,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )
    padded_timestamps = tf.squeeze(padded_timestamps)
    
    attention_mask = tf.cast(padded_packets[..., 0] != 0, dtype=tf.float32)
    
    length = int(np.sum(attention_mask))
    
    return padded_packets, padded_timestamps, attention_mask, length

def interpolate_packet(flow, t, m, max_num_packets, length):
    """Randomly interpolates between two consecutive packets in the flow."""
    index = random.randint(1, length - 2)

    interpolated_packet = (flow[index] + flow[index+1]) / 2
    padded_packets = tf.keras.utils.pad_sequences(
        [np.insert(flow, index, interpolated_packet, axis=0)],
        maxlen=max_num_packets,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )
    padded_packets = tf.squeeze(padded_packets)

    timestamp = (t[index] + t[index+1])/2
    padded_timestamps = tf.keras.utils.pad_sequences(
        [np.insert(t, index, timestamp, axis=0)],
        maxlen=max_num_packets,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )
    padded_timestamps = tf.squeeze(padded_timestamps)
    
    attention_mask = tf.cast(padded_packets[..., 0] != 0, dtype=tf.float32)
      
    length = int(np.sum(attention_mask))
    
    return padded_packets, padded_timestamps, attention_mask, length

def augment_flows(data, augmentations, max_num_packets, n=1, k=1):
    
    d, t, m = data
    
    augmented_d = []
    augmented_t = []
    augmented_m = []

    if augmentations == "rdi":
        possible_aug = [remove_packet, duplicate_packet, interpolate_packet]
    elif augmentations == "rd":
        possible_aug = [remove_packet, duplicate_packet]
    elif augmentations == "ri":
        possible_aug = [remove_packet, interpolate_packet]
    elif augmentations == "di":
        possible_aug = [duplicate_packet, interpolate_packet]
    elif augmentations == "r":
        possible_aug = [remove_packet]
    elif augmentations == "d":
        possible_aug = [duplicate_packet]
    elif augmentations == "i":
        possible_aug = [interpolate_packet]
    
    for d_, t_, m_ in zip(d, t, m): # max 3 times
        d_res, t_res, m_res = create_subflows(d_, t_, m_, k)

        for _ in range(n): # n times

            for d__, t__, m__ in zip(d_res, t_res, m_res): # multiple times

                length = int(np.sum(m__))

                if length <= 4: #ensure the flow has at least 5 packets
                    continue

                x1 = 5
                y1 = 2
                x2 = 300
                y2 = 100
                a = (y2-y1)/(x2-x1)
                b = (y1*x2-x1*y2)/(x2-x1)
                max_augmentations = int(a*length+b)
                num_augmentations = random.randint(0, max_augmentations)
                
                if length <= num_augmentations + 3: #ensure the flow has at least the required number of packets + some extra
                    continue
                    
                if num_augmentations == 0:
                    continue

                # augment timestamps (t__)
                max_fraction = 0.3
                perturbed_times = t__.copy()
                for i in range(1, len(t__)):
                    time_diff = t__[i] - t__[i - 1]
                    if time_diff <= 0:
                        break
                    
                    max_perturbation = time_diff * max_fraction
                    perturbation = np.random.uniform(-max_perturbation, max_perturbation)
                    perturbed_times[i] += perturbation
                    perturbed_times[i] = max(perturbed_times[i], perturbed_times[i - 1] + 1e-8)
                    
                t__ = perturbed_times.copy()
                
                for _ in range(num_augmentations):
                    augmentation = random.choice(possible_aug)
                    d__, t__, m__, length = augmentation(d__, t__, m__, max_num_packets, length)
                
                augmented_d.append(d__)
                augmented_t.append(t__)
                augmented_m.append(m__)
                
    final_d = np.array(augmented_d)
    final_t = np.array(augmented_t)
    final_m = np.array(augmented_m)
    
    return final_d, final_t, final_m

def augment_flows_v2(data, augmentations, max_num_packets, n=1, k=1):
    
    d, t, m = data
    
    augmented_d = []
    augmented_t = []
    augmented_m = []

    if augmentations == "rdi":
        possible_aug = [remove_packet, duplicate_packet, interpolate_packet]
    elif augmentations == "rd":
        possible_aug = [remove_packet, duplicate_packet]
    elif augmentations == "ri":
        possible_aug = [remove_packet, interpolate_packet]
    elif augmentations == "di":
        possible_aug = [duplicate_packet, interpolate_packet]
    elif augmentations == "r":
        possible_aug = [remove_packet]
    elif augmentations == "d":
        possible_aug = [duplicate_packet]
    elif augmentations == "i":
        possible_aug = [interpolate_packet]
    
    for d_, t_, m_ in zip(d, t, m):
    
        d_res, t_res, m_res = create_subflows(d_, t_, m_, k)

        for d__, t__, m__ in zip(d_res, t_res, m_res): # multiple times

            length = int(np.sum(m__))

            if length < 5: # ensure the flow has at least 5 packets for augmentation
                
                augmented_d.append(d__)
                augmented_t.append(t__)
                augmented_m.append(m__)
                
            else: # length >= 5

                for _ in range(n): # augment n times
                
                    d_curr = d__.copy()
                    t_curr = t__.copy()
                    m_curr = m__.copy()
                    
                    x1 = 5
                    y1 = 2
                    x2 = 300
                    y2 = 200
                    a = (y2-y1)/(x2-x1)
                    b = (y1*x2-x1*y2)/(x2-x1)
                    max_augmentations = int(a*length+b)
                    num_augmentations = random.randint(0, max_augmentations)
                    
                    if num_augmentations == 0:
                        continue

                    # augment timestamps (t_curr)
                    max_fraction = 0.5
                    perturbed_times = t_curr.copy()
                    for i in range(1, len(t_curr)):
                        
                        if i == len(t_curr)-1:
                            time_diff = t_curr[i]-t_curr[i-1]
                        else:
                            time_diff = min(t_curr[i]-t_curr[i-1], t_curr[i+1]-t_curr[i])
                        
                        if time_diff <= 0:
                            break
                        
                        max_perturbation = time_diff * max_fraction
                        perturbation = np.random.uniform(-max_perturbation, max_perturbation)
                        perturbed_times[i] += perturbation
                        perturbed_times[i] = max(perturbed_times[i], perturbed_times[i-1] + 1e-8)
                    
                    t_curr = perturbed_times.copy()
                
                    for _ in range(num_augmentations):
                        augmentation = random.choice(possible_aug)
                        d_curr, t_curr, m_curr, length = augmentation(d_curr, t_curr, m_curr, max_num_packets, length)
                        if length <= 5:
                            break
                
                    augmented_d.append(d_curr)
                    augmented_t.append(t_curr)
                    augmented_m.append(m_curr)
                
    final_d = np.array(augmented_d)
    final_t = np.array(augmented_t)
    final_m = np.array(augmented_m)
    
    return final_d, final_t, final_m

def augment_timestamps(data, n=1):
    
    d, t, m = data
    
    augmented_d = []
    augmented_t = []
    augmented_m = []
    
    for _ in range(n): # n times
    
        for d_, t_, m_ in zip(d, t, m):
        
            augmented_d.append(d_)
            augmented_m.append(m_)

            # augment timestamps (t_)
            max_fraction = 0.5
            perturbed_times = t_.copy()
            for i in range(1, len(t_)):
            
                if i == len(t_)-1:
                    time_diff = t_[i]-t_[i-1]
                else:
                    time_diff = min(t_[i]-t_[i-1], t_[i+1]-t_[i])
                
                if time_diff <= 0:
                    break
                    
                max_perturbation = time_diff * max_fraction
                perturbation = np.random.uniform(-max_perturbation, max_perturbation)
                perturbed_times[i] += perturbation
                perturbed_times[i] = max(perturbed_times[i], perturbed_times[i-1] + 1e-8)
            
            augmented_t.append(perturbed_times)
            
    final_d = np.array(augmented_d)
    final_t = np.array(augmented_t)
    final_m = np.array(augmented_m)
    
    return final_d, final_t, final_m

def randomly_keep_elements(d, t, m, length):
    num_elements = d.shape[0]
    indices = np.random.choice(num_elements, size=length, replace=False)
    return d[indices], t[indices], m[indices]

def read_files(root_dir):

    # benign
    
    # flow 1
    packets1 = rdpcap(root_dir+"filtered_benign_25.pcap")
    packets1.extend(rdpcap(root_dir+"filtered_benign1_40.pcap"))
    packets1.extend(rdpcap(root_dir+"filtered_benign3_4.pcap"))
    
    # flow 2
    packets2 = rdpcap(root_dir+"filtered_benign_87.pcap")
    
    # flow 3
    packets3 = rdpcap(root_dir+"filtered_benign2_10.pcap")
    packets3.extend(rdpcap(root_dir+"filtered_benign2_17.pcap"))
    packets3.extend(rdpcap(root_dir+"filtered_benign2_20.pcap"))
    
    flows_benign = {
        "1": packets1,
        "2": packets2,
        "3": packets3,
    }

    # attacks
    attackers = ["dc:a6:32:dc:27:d5"]
    victims = ["dc:a6:32:c9:e6:f4", "dc:a6:32:c9:e4:c6", "dc:a6:32:c9:e5:02"]

    pcap_file = root_dir+"SqlInjection_new.pcap"
    flows_sql = create_flows(pcap_file, attackers, victims)
    
    pcap_file = root_dir+"CommandInjection_new.pcap"
    flows_command = create_flows(pcap_file, attackers, victims)
    
    pcap_file = root_dir+"Backdoor_Malware_new.pcap"
    flows_backdoor = create_flows(pcap_file, attackers, victims)
    
    pcap_file = root_dir+"Uploading_Attack_new.pcap"
    flows_uploading = create_flows(pcap_file, attackers, victims)
    
    pcap_file = root_dir+"XSS_new.pcap"
    flows_xss = create_flows(pcap_file, attackers, victims)
    
    pcap_file = root_dir+"BrowserHijacking_new.pcap"
    flows_high = create_flows(pcap_file, attackers, victims)

    return flows_sql, flows_command, flows_backdoor, flows_uploading, flows_xss, flows_high, flows_benign
    
def read_files_v2(root_dir, max_num_packets):

    # benign
    #packets1 = rdpcap("/kaggle/input/pcap-files/filtered_benign2_113.pcap")
    #packets1 = packets1[:30]
    packets1 = rdpcap("/kaggle/input/pcap-files/filtered_benign2_113.pcap", count=max_num_packets)

    packets2 = rdpcap("/kaggle/input/pcap-files/filtered_benign_0.pcap", count=max_num_packets)
    #packets2 = packets2[:30]

    packets3 = rdpcap("/kaggle/input/pcap-files/filtered_benign_87.pcap", count=max_num_packets)
    #packets3 = packets3[:30]

    flows_benign = {
        "1": packets1,
        "2": packets2,
        "3": packets3,
    }

    # attacks
    attackers = ["dc:a6:32:dc:27:d5"]
    victims = ["dc:a6:32:c9:e6:f4", "dc:a6:32:c9:e4:c6", "dc:a6:32:c9:e5:02"]

    pcap_file = root_dir+"SqlInjection_new.pcap"
    flows_sql = create_flows(pcap_file, attackers, victims)
    
    pcap_file = root_dir+"CommandInjection_new.pcap"
    flows_command = create_flows(pcap_file, attackers, victims)
    
    pcap_file = root_dir+"Backdoor_Malware_new.pcap"
    flows_backdoor = create_flows(pcap_file, attackers, victims)
    
    pcap_file = root_dir+"Uploading_Attack_new.pcap"
    flows_uploading = create_flows(pcap_file, attackers, victims)
    
    pcap_file = root_dir+"XSS_new.pcap"
    flows_xss = create_flows(pcap_file, attackers, victims)
    
    pcap_file = root_dir+"BrowserHijacking_new.pcap"
    flows_high = create_flows(pcap_file, attackers, victims)

    return flows_sql, flows_command, flows_backdoor, flows_uploading, flows_xss, flows_high, flows_benign

def construct_dataset(root_dir, packet_length, max_num_packets, augmentations, augmentation_level, num_classes, split):

    start_time = time.time()
    print("reading pcap files and creating flows...")
    
    flows_sql, flows_command, flows_backdoor, flows_uploading, flows_xss, flows_high, flows_benign = read_files_v2(root_dir, max_num_packets)
    
    # preprocess
    print("preprocessing flows...")
    
    data_sql = trim_or_pad(flows_sql, packet_length, max_num_packets)
    data_command = trim_or_pad(flows_command, packet_length, max_num_packets)
    data_backdoor = trim_or_pad(flows_backdoor, packet_length, max_num_packets)
    data_uploading = trim_or_pad(flows_uploading, packet_length, max_num_packets)
    data_xss = trim_or_pad(flows_xss, packet_length, max_num_packets)
    if num_classes == 7:
        data_high = trim_or_pad(flows_high, packet_length, max_num_packets)
    data_benign = trim_or_pad(flows_benign, packet_length, max_num_packets)

    d_sql, t_sql, m_sql = create_masks(data_sql, max_num_packets)
    d_command, t_command, m_command = create_masks(data_command, max_num_packets)
    d_backdoor, t_backdoor, m_backdoor = create_masks(data_backdoor, max_num_packets)
    d_uploading, t_uploading, m_uploading = create_masks(data_uploading, max_num_packets)
    d_xss, t_xss, m_xss = create_masks(data_xss, max_num_packets)
    if num_classes == 7:
        d_high, t_high, m_high = create_masks(data_high, max_num_packets)
    d_benign, t_benign, m_benign = create_masks(data_benign, max_num_packets)

    # augmentation
    print("augmenting flows...")
    
    if augmentation_level == "low":
        if max_num_packets == 30:
            n = [20, 20, 20, 20, 20, 20, 20]
        elif max_num_packets == 50:
            n = [5, 5, 5, 9, 5, 5, 5]
        elif max_num_packets == 100:
            n = [2, 3, 3, 9, 5, 2, 2]
        elif max_num_packets == 200:
            n = [2, 3, 3, 9, 5, 1, 1]
    
    if augmentation_level == "mid":
        if max_num_packets == 30:
            n = [29, 29, 29, 29, 29, 29, 29]
    
    if augmentation_level == "high":
        if max_num_packets == 30:
            n = [39, 39, 39, 39, 39, 39, 39]
    
    # if augmentation_level == "mid":
    #     n = [x*2 for x  in n]
    # if augmentation_level == "high":
    #     n = [x*4 for x  in n]

    if split == "A":
        d_sql_, t_sql_, m_sql_ = d_sql[:2], t_sql[:2], m_sql[:2]
        d_command_, t_command_, m_command_ = d_command[:2], t_command[:2], m_command[:2]
        d_backdoor_, t_backdoor_, m_backdoor_ = d_backdoor[:2], t_backdoor[:2], m_backdoor[:2]
        d_uploading_, t_uploading_, m_uploading_ = d_uploading[:2], t_uploading[:2], m_uploading[:2]
        d_xss_, t_xss_, m_xss_ = d_xss[:2], t_xss[:2], m_xss[:2]
        if num_classes == 7:
            d_high_, t_high_, m_high_ = d_high[:2], t_high[:2], m_high[:2]
        d_benign_, t_benign_, m_benign_ = d_benign[:2], t_benign[:2], m_benign[:2]
    elif split == "B":
        d_sql_, t_sql_, m_sql_ = d_sql[1:], t_sql[1:], m_sql[1:]
        d_command_, t_command_, m_command_ = d_command[1:], t_command[1:], m_command[1:]
        d_backdoor_, t_backdoor_, m_backdoor_ = d_backdoor[1:], t_backdoor[1:], m_backdoor[1:]
        d_uploading_, t_uploading_, m_uploading_ = d_uploading[1:], t_uploading[1:], m_uploading[1:]
        d_xss_, t_xss_, m_xss_ = d_xss[1:], t_xss[1:], m_xss[1:]
        if num_classes == 7:
            d_high_, t_high_, m_high_ = d_high[1:], t_high[1:], m_high[1:]
        d_benign_, t_benign_, m_benign_ = d_benign[1:], t_benign[1:], m_benign[1:]
    elif split == "C":
        d_sql_, t_sql_, m_sql_ = d_sql[::2], t_sql[::2], m_sql[::2]
        d_command_, t_command_, m_command_ = d_command[::2], t_command[::2], m_command[::2]
        d_backdoor_, t_backdoor_, m_backdoor_ = d_backdoor[::2], t_backdoor[::2], m_backdoor[::2]
        d_uploading_, t_uploading_, m_uploading_ = d_uploading[::2], t_uploading[::2], m_uploading[::2]
        d_xss_, t_xss_, m_xss_ = d_xss[::2], t_xss[::2], m_xss[::2]
        if num_classes == 7:
            d_high_, t_high_, m_high_ = d_high[::2], t_high[::2], m_high[::2]
        d_benign_, t_benign_, m_benign_ = d_benign[::2], t_benign[::2], m_benign[::2]

    # number of packets for each flow in class 0: 132, 331, 253
    d_sql_aug, t_sql_aug, m_sql_aug = augment_flows((d_sql_, t_sql_, m_sql_), augmentations, max_num_packets, n=n[0], k=1)
    # number of packets for each flow in class 1: 75, 72, 86
    d_command_aug, t_command_aug, m_command_aug = augment_flows((d_command_, t_command_, m_command_), augmentations, max_num_packets, n=n[1], k=1)
    # number of packets for each flow in class 2: 86, 94, 84
    d_backdoor_aug, t_backdoor_aug, m_backdoor_aug = augment_flows((d_backdoor_, t_backdoor_, m_backdoor_), augmentations, max_num_packets, n=n[2], k=1)
    # number of packets for each flow in class 3: 28, 28, 29
    d_uploading_aug, t_uploading_aug, m_uploading_aug = augment_flows((d_uploading_, t_uploading_, m_uploading_), augmentations, max_num_packets, n=n[3], k=1)
    # number of packets for each flow in class 4: 50, 55, 40
    d_xss_aug, t_xss_aug, m_xss_aug = augment_flows((d_xss_, t_xss_, m_xss_), augmentations, max_num_packets, n=n[4], k=1)
    if num_classes == 7:
        # number of packets for each flow in class 5: 5042, 1006, 1907
        d_high_aug, t_high_aug, m_high_aug = augment_flows((d_high_, t_high_, m_high_), augmentations, max_num_packets, n=n[5], k=1)
    # number of packets for each flow in class 6: 198, 900, 206
    d_benign_aug, t_benign_aug, m_benign_aug = augment_flows((d_benign_, t_benign_, m_benign_), augmentations, max_num_packets, n=n[6], k=1)
    
    if num_classes == 7:
        min_number = np.min([d_sql_aug.shape[0], d_command_aug.shape[0], d_backdoor_aug.shape[0], d_uploading_aug.shape[0], d_xss_aug.shape[0], d_high_aug.shape[0], d_benign_aug.shape[0]])
    else:
        min_number = np.min([d_sql_aug.shape[0], d_command_aug.shape[0], d_backdoor_aug.shape[0], d_uploading_aug.shape[0], d_xss_aug.shape[0], d_benign_aug.shape[0]])
    
    d_sql_aug, t_sql_aug, m_sql_aug = randomly_keep_elements(d_sql_aug, t_sql_aug, m_sql_aug, min_number)
    d_command_aug, t_command_aug, m_command_aug = randomly_keep_elements(d_command_aug, t_command_aug, m_command_aug, min_number)
    d_backdoor_aug, t_backdoor_aug, m_backdoor_aug = randomly_keep_elements(d_backdoor_aug, t_backdoor_aug, m_backdoor_aug, min_number)
    d_uploading_aug, t_uploading_aug, m_uploading_aug = randomly_keep_elements(d_uploading_aug, t_uploading_aug, m_uploading_aug, min_number)
    d_xss_aug, t_xss_aug, m_xss_aug = randomly_keep_elements(d_xss_aug, t_xss_aug, m_xss_aug, min_number)
    if num_classes == 7:
        d_high_aug, t_high_aug, m_high_aug = randomly_keep_elements(d_high_aug, t_high_aug, m_high_aug, min_number)
    d_benign_aug, t_benign_aug, m_benign_aug = randomly_keep_elements(d_benign_aug, t_benign_aug, m_benign_aug, min_number)

    # training, val, test split
    print("preparing training, val and test sets...")
    
    if num_classes == 7:
        x_train = np.concatenate((d_sql_aug, d_command_aug, d_backdoor_aug, d_uploading_aug, d_xss_aug, d_high_aug, d_benign_aug), axis=0)
        t_train = np.concatenate((t_sql_aug, t_command_aug, t_backdoor_aug, t_uploading_aug, t_xss_aug, t_high_aug, t_benign_aug), axis=0)
        m_train = np.concatenate((m_sql_aug, m_command_aug, m_backdoor_aug, m_uploading_aug, m_xss_aug, m_high_aug, m_benign_aug), axis=0)
        y = d_sql_aug.shape[0]*[0] + d_command_aug.shape[0]*[1] + d_backdoor_aug.shape[0]*[2] + d_uploading_aug.shape[0]*[3] + d_xss_aug.shape[0]*[4] + d_high_aug.shape[0]*[5] + d_benign_aug.shape[0]*[6]
    else:
        x_train = np.concatenate((d_sql_aug, d_command_aug, d_backdoor_aug, d_uploading_aug, d_xss_aug, d_benign_aug), axis=0)
        t_train = np.concatenate((t_sql_aug, t_command_aug, t_backdoor_aug, t_uploading_aug, t_xss_aug, t_benign_aug), axis=0)
        m_train = np.concatenate((m_sql_aug, m_command_aug, m_backdoor_aug, m_uploading_aug, m_xss_aug, m_benign_aug), axis=0)
        y = d_sql_aug.shape[0]*[0] + d_command_aug.shape[0]*[1] + d_backdoor_aug.shape[0]*[2] + d_uploading_aug.shape[0]*[3] + d_xss_aug.shape[0]*[4] + d_benign_aug.shape[0]*[5]

    y_train = np.array(y)

    x_train, x_val, t_train, t_val, m_train, m_val, y_train, y_val = train_test_split(x_train, t_train, m_train, y_train, test_size=0.05, random_state=42)

    if split == "A":
        d_sql_test, t_sql_test, m_sql_test = d_sql[2:], t_sql[2:], m_sql[2:]
        d_command_test, t_command_test, m_command_test = d_command[2:], t_command[2:], m_command[2:]
        d_backdoor_test, t_backdoor_test, m_backdoor_test = d_backdoor[2:], t_backdoor[2:], m_backdoor[2:]
        d_uploading_test, t_uploading_test, m_uploading_test = d_uploading[2:], t_uploading[2:], m_uploading[2:]
        d_xss_test, t_xss_test, m_xss_test = d_xss[2:], t_xss[2:], m_xss[2:]
        if num_classes == 7:
            d_high_test, t_high_test, m_high_test = d_high[2:], t_high[2:], m_high[2:]
        d_benign_test, t_benign_test, m_benign_test = d_benign[2:], t_benign[2:], m_benign[2:]
    elif split == "B":
        d_sql_test, t_sql_test, m_sql_test = d_sql[0:1], t_sql[0:1], m_sql[0:1]
        d_command_test, t_command_test, m_command_test = d_command[0:1], t_command[0:1], m_command[0:1]
        d_backdoor_test, t_backdoor_test, m_backdoor_test = d_backdoor[0:1], t_backdoor[0:1], m_backdoor[0:1]
        d_uploading_test, t_uploading_test, m_uploading_test = d_uploading[0:1], t_uploading[0:1], m_uploading[0:1]
        d_xss_test, t_xss_test, m_xss_test = d_xss[0:1], t_xss[0:1], m_xss[0:1]
        if num_classes == 7:
            d_high_test, t_high_test, m_high_test = d_high[0:1], t_high[0:1], m_high[0:1]
        d_benign_test, t_benign_test, m_benign_test = d_benign[0:1], t_benign[0:1], m_benign[0:1]
    elif split == "C":
        d_sql_test, t_sql_test, m_sql_test = d_sql[1:2], t_sql[1:2], m_sql[1:2]
        d_command_test, t_command_test, m_command_test = d_command[1:2], t_command[1:2], m_command[1:2]
        d_backdoor_test, t_backdoor_test, m_backdoor_test = d_backdoor[1:2], t_backdoor[1:2], m_backdoor[1:2]
        d_uploading_test, t_uploading_test, m_uploading_test = d_uploading[1:2], t_uploading[1:2], m_uploading[1:2]
        d_xss_test, t_xss_test, m_xss_test = d_xss[1:2], t_xss[1:2], m_xss[1:2]
        if num_classes == 7:
            d_high_test, t_high_test, m_high_test = d_high[1:2], t_high[1:2], m_high[1:2]
        d_benign_test, t_benign_test, m_benign_test = d_benign[1:2], t_benign[1:2], m_benign[1:2]
    
    if num_classes == 7:
        x_test = np.concatenate((d_sql_test, d_command_test, d_backdoor_test, d_uploading_test, d_xss_test, d_high_test, d_benign_test), axis=0)
        t_test = np.concatenate((t_sql_test, t_command_test, t_backdoor_test, t_uploading_test, t_xss_test, t_high_test, t_benign_test), axis=0)
        m_test = np.concatenate((m_sql_test, m_command_test, m_backdoor_test, m_uploading_test, m_xss_test, m_high_test, m_benign_test), axis=0)
        y = d_sql_test.shape[0]*[0] + d_command_test.shape[0]*[1] + d_backdoor_test.shape[0]*[2] + d_uploading_test.shape[0]*[3] + d_xss_test.shape[0]*[4] + d_high_test.shape[0]*[5] + d_benign_test.shape[0]*[6]
    else:
        x_test = np.concatenate((d_sql_test, d_command_test, d_backdoor_test, d_uploading_test, d_xss_test, d_benign_test), axis=0)
        t_test = np.concatenate((t_sql_test, t_command_test, t_backdoor_test, t_uploading_test, t_xss_test, t_benign_test), axis=0)
        m_test = np.concatenate((m_sql_test, m_command_test, m_backdoor_test, m_uploading_test, m_xss_test, m_benign_test), axis=0)
        y = d_sql_test.shape[0]*[0] + d_command_test.shape[0]*[1] + d_backdoor_test.shape[0]*[2] + d_uploading_test.shape[0]*[3] + d_xss_test.shape[0]*[4] + d_benign_test.shape[0]*[5]

    y_test = np.array(y)

    print(f"- Split {split}")
    print(f"- Training samples: {x_train.shape[0]}")
    print(f"- Validation samples: {x_val.shape[0]}")
    print(f"- Testing samples: {x_test.shape[0]}")

    # TF dataset
    BATCH_SIZE = 8
    BUFFER_SIZE = BATCH_SIZE * 2
    AUTO = tf.data.AUTOTUNE
    
    train_ds = tf.data.Dataset.from_tensor_slices(((x_train, t_train, m_train), y_train))
    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

    val_ds = tf.data.Dataset.from_tensor_slices(((x_val, t_val, m_val), y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTO)

    end_time = time.time()
    execution_time = int(end_time - start_time)
    print(f"dataset created succesfully in {execution_time} seconds!")
    
    return train_ds, val_ds, (x_test, t_test, m_test, y_test)

def prepare_dataset(data, packet_length, max_num_packets, augmentations, augmentation_level):

    start_time = time.time()
    print("reading pcap files and creating flows...")
    
    flows_sql, flows_command, flows_backdoor, flows_uploading, flows_xss, flows_high, flows_benign = data
    
    # preprocess
    print("preprocessing flows...")
    
    data_sql = trim_or_pad(flows_sql, packet_length, max_num_packets)
    data_command = trim_or_pad(flows_command, packet_length, max_num_packets)
    data_backdoor = trim_or_pad(flows_backdoor, packet_length, max_num_packets)
    data_uploading = trim_or_pad(flows_uploading, packet_length, max_num_packets)
    data_xss = trim_or_pad(flows_xss, packet_length, max_num_packets)
    data_high = trim_or_pad(flows_high, packet_length, max_num_packets)
    data_benign = trim_or_pad(flows_benign, packet_length, max_num_packets)

    d_sql, t_sql, m_sql = create_masks(data_sql, max_num_packets)
    d_command, t_command, m_command = create_masks(data_command, max_num_packets)
    d_backdoor, t_backdoor, m_backdoor = create_masks(data_backdoor, max_num_packets)
    d_uploading, t_uploading, m_uploading = create_masks(data_uploading, max_num_packets)
    d_xss, t_xss, m_xss = create_masks(data_xss, max_num_packets)
    d_high, t_high, m_high = create_masks(data_high, max_num_packets)
    d_benign, t_benign, m_benign = create_masks(data_benign, max_num_packets)

    # augmentation
    print("augmenting flows...")
    if max_num_packets == 30:
        n = [9, 9, 9, 9, 9, 9, 9]
    elif max_num_packets == 50:
        n = [5, 5, 5, 9, 5, 5, 5]
    elif max_num_packets == 100:
        n = [2, 3, 3, 9, 5, 2, 2]
    elif max_num_packets == 200:
        n = [2, 3, 3, 9, 5, 1, 1]
    
    if augmentation_level == "mid":
        n = [x*2 for x  in n]
    if augmentation_level == "high":
        n = [x*4 for x  in n]

    # # number of packets for each flow in class 0: 132, 331, 253
    # d_sql_aug, t_sql_aug, m_sql_aug = augment_flows((d_sql, t_sql, m_sql), augmentations, max_num_packets, n=n[0], k=1)
    # # number of packets for each flow in class 1: 75, 72, 86
    # d_command_aug, t_command_aug, m_command_aug = augment_flows((d_command, t_command, m_command), augmentations, max_num_packets, n=n[1], k=1)
    # # number of packets for each flow in class 2: 86, 94, 84
    # d_backdoor_aug, t_backdoor_aug, m_backdoor_aug = augment_flows((d_backdoor, t_backdoor, m_backdoor), augmentations, max_num_packets, n=n[2], k=1)
    # # number of packets for each flow in class 3: 28, 28, 29
    # d_uploading_aug, t_uploading_aug, m_uploading_aug = augment_flows((d_uploading, t_uploading, m_uploading), augmentations, max_num_packets, n=n[3], k=1)
    # # number of packets for each flow in class 4: 50, 55, 40
    # d_xss_aug, t_xss_aug, m_xss_aug = augment_flows((d_xss, t_xss, m_xss), augmentations, max_num_packets, n=n[4], k=1)
    # # number of packets for each flow in class 5: 5042, 1006, 1907
    # d_high_aug, t_high_aug, m_high_aug = augment_flows((d_high, t_high, m_high), augmentations, max_num_packets, n=n[5], k=1)
    # # number of packets for each flow in class 6: 198, 900, 206
    # d_benign_aug, t_benign_aug, m_benign_aug = augment_flows((d_benign, t_benign, m_benign), augmentations, max_num_packets, n=n[6], k=1)

    # number of packets for each flow in class 0: 132, 331, 253
    d_sql_aug, t_sql_aug, m_sql_aug = augment_flows((d_sql[:2], t_sql[:2], m_sql[:2]), augmentations, max_num_packets, n=n[0], k=1)
    # number of packets for each flow in class 1: 75, 72, 86
    d_command_aug, t_command_aug, m_command_aug = augment_flows((d_command[:2], t_command[:2], m_command[:2]), augmentations, max_num_packets, n=n[1], k=1)
    # number of packets for each flow in class 2: 86, 94, 84
    d_backdoor_aug, t_backdoor_aug, m_backdoor_aug = augment_flows((d_backdoor[:2], t_backdoor[:2], m_backdoor[:2]), augmentations, max_num_packets, n=n[2], k=1)
    # number of packets for each flow in class 3: 28, 28, 29
    d_uploading_aug, t_uploading_aug, m_uploading_aug = augment_flows((d_uploading[:2], t_uploading[:2], m_uploading[:2]), augmentations, max_num_packets, n=n[3], k=1)
    # number of packets for each flow in class 4: 50, 55, 40
    d_xss_aug, t_xss_aug, m_xss_aug = augment_flows((d_xss[:2], t_xss[:2], m_xss[:2]), augmentations, max_num_packets, n=n[4], k=1)
    # number of packets for each flow in class 5: 5042, 1006, 1907
    d_high_aug, t_high_aug, m_high_aug = augment_flows((d_high[:2], t_high[:2], m_high[:2]), augmentations, max_num_packets, n=n[5], k=1)
    # number of packets for each flow in class 6: 198, 900, 206
    d_benign_aug, t_benign_aug, m_benign_aug = augment_flows((d_benign[:2], t_benign[:2], m_benign[:2]), augmentations, max_num_packets, n=n[6], k=1)
    
    min_number = np.min([d_sql_aug.shape[0], d_command_aug.shape[0], d_backdoor_aug.shape[0], d_uploading_aug.shape[0], d_xss_aug.shape[0], d_high_aug.shape[0], d_benign_aug.shape[0]])

    d_sql_aug, t_sql_aug, m_sql_aug = randomly_keep_elements(d_sql_aug, t_sql_aug, m_sql_aug, min_number)
    d_command_aug, t_command_aug, m_command_aug = randomly_keep_elements(d_command_aug, t_command_aug, m_command_aug, min_number)
    d_backdoor_aug, t_backdoor_aug, m_backdoor_aug = randomly_keep_elements(d_backdoor_aug, t_backdoor_aug, m_backdoor_aug, min_number)
    d_uploading_aug, t_uploading_aug, m_uploading_aug = randomly_keep_elements(d_uploading_aug, t_uploading_aug, m_uploading_aug, min_number)
    d_xss_aug, t_xss_aug, m_xss_aug = randomly_keep_elements(d_xss_aug, t_xss_aug, m_xss_aug, min_number)
    d_high_aug, t_high_aug, m_high_aug = randomly_keep_elements(d_high_aug, t_high_aug, m_high_aug, min_number)
    d_benign_aug, t_benign_aug, m_benign_aug = randomly_keep_elements(d_benign_aug, t_benign_aug, m_benign_aug, min_number)

    # training, val, test split
    print("preparing training, val and test sets...")
    
    x_train = np.concatenate((d_sql_aug, d_command_aug, d_backdoor_aug, d_uploading_aug, d_xss_aug, d_high_aug, d_benign_aug), axis=0)
    t_train = np.concatenate((t_sql_aug, t_command_aug, t_backdoor_aug, t_uploading_aug, t_xss_aug, t_high_aug, t_benign_aug), axis=0)
    m_train = np.concatenate((m_sql_aug, m_command_aug, m_backdoor_aug, m_uploading_aug, m_xss_aug, m_high_aug, m_benign_aug), axis=0)
    y = d_sql_aug.shape[0]*[0] + d_command_aug.shape[0]*[1] + d_backdoor_aug.shape[0]*[2] + d_uploading_aug.shape[0]*[3] + d_xss_aug.shape[0]*[4] + d_high_aug.shape[0]*[5] + d_benign_aug.shape[0]*[6]
    y_train = np.array(y)

    x_train, x_val, t_train, t_val, m_train, m_val, y_train, y_val = train_test_split(x_train, t_train, m_train, y_train, test_size=0.05, random_state=42)

    # x_test = np.concatenate((d_sql, d_command, d_backdoor, d_uploading, d_xss, d_high, d_benign), axis=0)
    # t_test = np.concatenate((t_sql, t_command, t_backdoor, t_uploading, t_xss, t_high, t_benign), axis=0)
    # m_test = np.concatenate((m_sql, m_command, m_backdoor, m_uploading, m_xss, m_high, m_benign), axis=0)
    # y = 3*[0] + 3*[1] + 3*[2] + 3*[3] + 3*[4] + 3*[5] + 3*[6]
    # y_test = np.array(y)
    
    x_test = np.concatenate((d_sql[2:], d_command[2:], d_backdoor[2:], d_uploading[2:], d_xss[2:], d_high[2:], d_benign[2:]), axis=0)
    t_test = np.concatenate((t_sql[2:], t_command[2:], t_backdoor[2:], t_uploading[2:], t_xss[2:], t_high[2:], t_benign[2:]), axis=0)
    m_test = np.concatenate((m_sql[2:], m_command[2:], m_backdoor[2:], m_uploading[2:], m_xss[2:], m_high[2:], m_benign[2:]), axis=0)
    y = [0, 1, 2, 3, 4, 5, 6]
    y_test = np.array(y)

    print(f"- Training samples: {x_train.shape[0]}")
    print(f"- Validation samples: {x_val.shape[0]}")
    print(f"- Testing samples: {x_test.shape[0]}")

    # TF dataset
    BATCH_SIZE = 8
    BUFFER_SIZE = BATCH_SIZE * 2
    AUTO = tf.data.AUTOTUNE
    
    train_ds = tf.data.Dataset.from_tensor_slices(((x_train, t_train, m_train), y_train))
    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

    val_ds = tf.data.Dataset.from_tensor_slices(((x_val, t_val, m_val), y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTO)

    end_time = time.time()
    execution_time = int(end_time - start_time)
    print(f"dataset created succesfully in {execution_time} seconds!")
    
    return train_ds, val_ds, (x_test, t_test, m_test, y_test)

def prepare_dataset_v2(data, packet_length, max_num_packets, augmentations, augmentation_level):

    start_time = time.time()
    print("reading pcap files and creating flows...")
    
    flows_sql, flows_command, flows_backdoor, flows_uploading, flows_xss, flows_high, flows_benign = data
    
    # preprocess
    print("preprocessing flows...")
    
    data_sql = trim_or_pad(flows_sql, packet_length, max_num_packets)
    data_command = trim_or_pad(flows_command, packet_length, max_num_packets)
    data_backdoor = trim_or_pad(flows_backdoor, packet_length, max_num_packets)
    data_uploading = trim_or_pad(flows_uploading, packet_length, max_num_packets)
    data_xss = trim_or_pad(flows_xss, packet_length, max_num_packets)
    data_high = trim_or_pad(flows_high, packet_length, max_num_packets)
    data_benign = trim_or_pad(flows_benign, packet_length, max_num_packets)

    d_sql, t_sql, m_sql = create_masks(data_sql, max_num_packets)
    d_command, t_command, m_command = create_masks(data_command, max_num_packets)
    d_backdoor, t_backdoor, m_backdoor = create_masks(data_backdoor, max_num_packets)
    d_uploading, t_uploading, m_uploading = create_masks(data_uploading, max_num_packets)
    d_xss, t_xss, m_xss = create_masks(data_xss, max_num_packets)
    d_high, t_high, m_high = create_masks(data_high, max_num_packets)
    d_benign, t_benign, m_benign = create_masks(data_benign, max_num_packets)

    # augmentation
    print("augmenting flows...")
    if max_num_packets == 30:
        n = [9, 9, 9, 9, 9, 9, 9]
    elif max_num_packets == 50:
        n = [5, 5, 5, 9, 5, 5, 5]
    elif max_num_packets == 100:
        n = [2, 3, 3, 9, 5, 2, 2]
    elif max_num_packets == 200:
        n = [2, 3, 3, 9, 5, 1, 1]
    
    if augmentation_level == "mid":
        n = [x*2 for x  in n]
    if augmentation_level == "high":
        n = [x*4 for x  in n]

    # # number of packets for each flow in class 0: 132, 331, 253
    # d_sql_aug, t_sql_aug, m_sql_aug = augment_flows((d_sql, t_sql, m_sql), augmentations, max_num_packets, n=n[0], k=1)
    # # number of packets for each flow in class 1: 75, 72, 86
    # d_command_aug, t_command_aug, m_command_aug = augment_flows((d_command, t_command, m_command), augmentations, max_num_packets, n=n[1], k=1)
    # # number of packets for each flow in class 2: 86, 94, 84
    # d_backdoor_aug, t_backdoor_aug, m_backdoor_aug = augment_flows((d_backdoor, t_backdoor, m_backdoor), augmentations, max_num_packets, n=n[2], k=1)
    # # number of packets for each flow in class 3: 28, 28, 29
    # d_uploading_aug, t_uploading_aug, m_uploading_aug = augment_flows((d_uploading, t_uploading, m_uploading), augmentations, max_num_packets, n=n[3], k=1)
    # # number of packets for each flow in class 4: 50, 55, 40
    # d_xss_aug, t_xss_aug, m_xss_aug = augment_flows((d_xss, t_xss, m_xss), augmentations, max_num_packets, n=n[4], k=1)
    # # number of packets for each flow in class 5: 5042, 1006, 1907
    # d_high_aug, t_high_aug, m_high_aug = augment_flows((d_high, t_high, m_high), augmentations, max_num_packets, n=n[5], k=1)
    # # number of packets for each flow in class 6: 198, 900, 206
    # d_benign_aug, t_benign_aug, m_benign_aug = augment_flows((d_benign, t_benign, m_benign), augmentations, max_num_packets, n=n[6], k=1)

    # number of packets for each flow in class 0: 132, 331, 253
    d_sql_aug, t_sql_aug, m_sql_aug = augment_flows((d_sql[:2], t_sql[:2], m_sql[:2]), augmentations, max_num_packets, n=n[0], k=1)
    # number of packets for each flow in class 1: 75, 72, 86
    d_command_aug, t_command_aug, m_command_aug = augment_flows((d_command[:2], t_command[:2], m_command[:2]), augmentations, max_num_packets, n=n[1], k=1)
    # number of packets for each flow in class 2: 86, 94, 84
    d_backdoor_aug, t_backdoor_aug, m_backdoor_aug = augment_flows((d_backdoor[:2], t_backdoor[:2], m_backdoor[:2]), augmentations, max_num_packets, n=n[2], k=1)
    # number of packets for each flow in class 3: 28, 28, 29
    d_uploading_aug, t_uploading_aug, m_uploading_aug = augment_flows((d_uploading[:2], t_uploading[:2], m_uploading[:2]), augmentations, max_num_packets, n=n[3], k=1)
    # number of packets for each flow in class 4: 50, 55, 40
    d_xss_aug, t_xss_aug, m_xss_aug = augment_flows((d_xss[:2], t_xss[:2], m_xss[:2]), augmentations, max_num_packets, n=n[4], k=1)
    # number of packets for each flow in class 5: 5042, 1006, 1907
    d_high_aug, t_high_aug, m_high_aug = augment_flows((d_high[:2], t_high[:2], m_high[:2]), augmentations, max_num_packets, n=n[5], k=1)
    # number of packets for each flow in class 6: 198, 900, 206
    d_benign_aug, t_benign_aug, m_benign_aug = augment_flows((d_benign[:2], t_benign[:2], m_benign[:2]), augmentations, max_num_packets, n=n[6], k=1)
    
    min_number = np.min([d_sql_aug.shape[0], d_command_aug.shape[0], d_backdoor_aug.shape[0], d_uploading_aug.shape[0], d_xss_aug.shape[0], d_high_aug.shape[0], d_benign_aug.shape[0]])

    d_sql_aug, t_sql_aug, m_sql_aug = randomly_keep_elements(d_sql_aug, t_sql_aug, m_sql_aug, min_number)
    d_command_aug, t_command_aug, m_command_aug = randomly_keep_elements(d_command_aug, t_command_aug, m_command_aug, min_number)
    d_backdoor_aug, t_backdoor_aug, m_backdoor_aug = randomly_keep_elements(d_backdoor_aug, t_backdoor_aug, m_backdoor_aug, min_number)
    d_uploading_aug, t_uploading_aug, m_uploading_aug = randomly_keep_elements(d_uploading_aug, t_uploading_aug, m_uploading_aug, min_number)
    d_xss_aug, t_xss_aug, m_xss_aug = randomly_keep_elements(d_xss_aug, t_xss_aug, m_xss_aug, min_number)
    d_high_aug, t_high_aug, m_high_aug = randomly_keep_elements(d_high_aug, t_high_aug, m_high_aug, min_number)
    d_benign_aug, t_benign_aug, m_benign_aug = randomly_keep_elements(d_benign_aug, t_benign_aug, m_benign_aug, min_number)

    # test
    d_sql_aug_t, t_sql_aug_t, m_sql_aug_t = augment_timestamps((d_sql[2:], t_sql[2:], m_sql[2:]), n=3)
    d_command_aug_t, t_command_aug_t, m_command_aug_t = augment_timestamps((d_command[2:], t_command[2:], m_command[2:]), n=3)
    d_backdoor_aug_t, t_backdoor_aug_t, m_backdoor_aug_t = augment_timestamps((d_backdoor[2:], t_backdoor[2:], m_backdoor[2:]), n=3)
    d_uploading_aug_t, t_uploading_aug_t, m_uploading_aug_t = augment_timestamps((d_uploading[2:], t_uploading[2:], m_uploading[2:]), n=3)
    d_xss_aug_t, t_xss_aug_t, m_xss_aug_t = augment_timestamps((d_xss[2:], t_xss[2:], m_xss[2:]), n=3)
    d_high_aug_t, t_high_aug_t, m_high_aug_t = augment_timestamps((d_high[2:], t_high[2:], m_high[2:]), n=3)
    d_benign_aug_t, t_benign_aug_t, m_benign_aug_t = augment_timestamps((d_benign[2:], t_benign[2:], m_benign[2:]), n=3)
    

    # training, val, test split
    print("preparing training, val and test sets...")
    
    x_train = np.concatenate((d_sql_aug, d_command_aug, d_backdoor_aug, d_uploading_aug, d_xss_aug, d_high_aug, d_benign_aug), axis=0)
    t_train = np.concatenate((t_sql_aug, t_command_aug, t_backdoor_aug, t_uploading_aug, t_xss_aug, t_high_aug, t_benign_aug), axis=0)
    m_train = np.concatenate((m_sql_aug, m_command_aug, m_backdoor_aug, m_uploading_aug, m_xss_aug, m_high_aug, m_benign_aug), axis=0)
    y = d_sql_aug.shape[0]*[0] + d_command_aug.shape[0]*[1] + d_backdoor_aug.shape[0]*[2] + d_uploading_aug.shape[0]*[3] + d_xss_aug.shape[0]*[4] + d_high_aug.shape[0]*[5] + d_benign_aug.shape[0]*[6]
    y_train = np.array(y)

    x_train, x_val, t_train, t_val, m_train, m_val, y_train, y_val = train_test_split(x_train, t_train, m_train, y_train, test_size=0.05, random_state=42)

    # x_test = np.concatenate((d_sql, d_command, d_backdoor, d_uploading, d_xss, d_high, d_benign), axis=0)
    # t_test = np.concatenate((t_sql, t_command, t_backdoor, t_uploading, t_xss, t_high, t_benign), axis=0)
    # m_test = np.concatenate((m_sql, m_command, m_backdoor, m_uploading, m_xss, m_high, m_benign), axis=0)
    # y = 3*[0] + 3*[1] + 3*[2] + 3*[3] + 3*[4] + 3*[5] + 3*[6]
    # y_test = np.array(y)
    
    x_test = np.concatenate((d_sql_aug_t, d_command_aug_t, d_backdoor_aug_t, d_uploading_aug_t, d_xss_aug_t, d_high_aug_t, d_benign_aug_t), axis=0)
    t_test = np.concatenate((t_sql_aug_t, t_command_aug_t, t_backdoor_aug_t, t_uploading_aug_t, t_xss_aug_t, t_high_aug_t, t_benign_aug_t), axis=0)
    m_test = np.concatenate((m_sql_aug_t, m_command_aug_t, m_backdoor_aug_t, m_uploading_aug_t, m_xss_aug_t, m_high_aug_t, m_benign_aug_t), axis=0)
    #y = [0, 1, 2, 3, 4, 5, 6]
    y = d_sql_aug_t.shape[0]*[0] + d_command_aug_t.shape[0]*[1] + d_backdoor_aug_t.shape[0]*[2] + d_uploading_aug_t.shape[0]*[3] + d_xss_aug_t.shape[0]*[4] + d_high_aug_t.shape[0]*[5] + d_benign_aug_t.shape[0]*[6]
    y_test = np.array(y)

    print(f"- Training samples: {x_train.shape[0]}")
    print(f"- Validation samples: {x_val.shape[0]}")
    print(f"- Testing samples: {x_test.shape[0]}")

    # TF dataset
    BATCH_SIZE = 8
    BUFFER_SIZE = BATCH_SIZE * 2
    AUTO = tf.data.AUTOTUNE
    
    train_ds = tf.data.Dataset.from_tensor_slices(((x_train, t_train, m_train), y_train))
    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

    val_ds = tf.data.Dataset.from_tensor_slices(((x_val, t_val, m_val), y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTO)

    end_time = time.time()
    execution_time = int(end_time - start_time)
    print(f"dataset created succesfully in {execution_time} seconds!")
    
    return train_ds, val_ds, (x_test, t_test, m_test, y_test)

def prepare_dataset_v3(data, packet_length, max_num_packets, augmentations, augmentation_level):

    start_time = time.time()
    print("reading pcap files and creating flows...")
    
    flows_sql, flows_command, flows_backdoor, flows_uploading, flows_xss, flows_high, flows_benign = data
    
    # preprocess
    print("preprocessing flows...")
    
    data_sql = trim_or_pad(flows_sql, packet_length, max_num_packets)
    data_command = trim_or_pad(flows_command, packet_length, max_num_packets)
    data_backdoor = trim_or_pad(flows_backdoor, packet_length, max_num_packets)
    data_uploading = trim_or_pad(flows_uploading, packet_length, max_num_packets)
    data_xss = trim_or_pad(flows_xss, packet_length, max_num_packets)
    data_high = trim_or_pad(flows_high, packet_length, max_num_packets)
    data_benign = trim_or_pad(flows_benign, packet_length, max_num_packets)

    d_sql, t_sql, m_sql = create_masks(data_sql, max_num_packets)
    d_command, t_command, m_command = create_masks(data_command, max_num_packets)
    d_backdoor, t_backdoor, m_backdoor = create_masks(data_backdoor, max_num_packets)
    d_uploading, t_uploading, m_uploading = create_masks(data_uploading, max_num_packets)
    d_xss, t_xss, m_xss = create_masks(data_xss, max_num_packets)
    d_high, t_high, m_high = create_masks(data_high, max_num_packets)
    d_benign, t_benign, m_benign = create_masks(data_benign, max_num_packets)

    # augmentation
    print("augmenting flows...")
    if max_num_packets == 30:
        n = [9, 9, 9, 9, 9, 9, 9]
    elif max_num_packets == 50:
        n = [5, 5, 5, 9, 5, 5, 5]
    elif max_num_packets == 100:
        n = [2, 3, 3, 9, 5, 2, 2]
    elif max_num_packets == 200:
        n = [2, 3, 3, 9, 5, 1, 1]
    
    if augmentation_level == "mid":
        n = [x*2 for x  in n]
    if augmentation_level == "high":
        n = [x*4 for x  in n]

    augment_fn = augment_flows
    # number of packets for each flow in class 0: 132, 331, 253
    d_sql_aug, t_sql_aug, m_sql_aug = augment_fn((d_sql[:2], t_sql[:2], m_sql[:2]), augmentations, max_num_packets, n=n[0], k=1)
    # number of packets for each flow in class 1: 75, 72, 86
    d_command_aug, t_command_aug, m_command_aug = augment_fn((d_command[:2], t_command[:2], m_command[:2]), augmentations, max_num_packets, n=n[1], k=1)
    # number of packets for each flow in class 2: 86, 94, 84
    d_backdoor_aug, t_backdoor_aug, m_backdoor_aug = augment_fn((d_backdoor[:2], t_backdoor[:2], m_backdoor[:2]), augmentations, max_num_packets, n=n[2], k=1)
    # number of packets for each flow in class 3: 28, 28, 29
    d_uploading_aug, t_uploading_aug, m_uploading_aug = augment_fn((d_uploading[:2], t_uploading[:2], m_uploading[:2]), augmentations, max_num_packets, n=n[3], k=1)
    # number of packets for each flow in class 4: 50, 55, 40
    d_xss_aug, t_xss_aug, m_xss_aug = augment_fn((d_xss[:2], t_xss[:2], m_xss[:2]), augmentations, max_num_packets, n=n[4], k=1)
    # number of packets for each flow in class 5: 5042, 1006, 1907
    d_high_aug, t_high_aug, m_high_aug = augment_fn((d_high[:2], t_high[:2], m_high[:2]), augmentations, max_num_packets, n=n[5], k=1)
    # number of packets for each flow in class 6: 198, 900, 206
    d_benign_aug, t_benign_aug, m_benign_aug = augment_fn((d_benign[:2], t_benign[:2], m_benign[:2]), augmentations, max_num_packets, n=n[6], k=1)
    
    min_number = np.min([d_sql_aug.shape[0], d_command_aug.shape[0], d_backdoor_aug.shape[0], d_uploading_aug.shape[0], d_xss_aug.shape[0], d_high_aug.shape[0], d_benign_aug.shape[0]])

    d_sql_aug, t_sql_aug, m_sql_aug = randomly_keep_elements(d_sql_aug, t_sql_aug, m_sql_aug, min_number)
    d_command_aug, t_command_aug, m_command_aug = randomly_keep_elements(d_command_aug, t_command_aug, m_command_aug, min_number)
    d_backdoor_aug, t_backdoor_aug, m_backdoor_aug = randomly_keep_elements(d_backdoor_aug, t_backdoor_aug, m_backdoor_aug, min_number)
    d_uploading_aug, t_uploading_aug, m_uploading_aug = randomly_keep_elements(d_uploading_aug, t_uploading_aug, m_uploading_aug, min_number)
    d_xss_aug, t_xss_aug, m_xss_aug = randomly_keep_elements(d_xss_aug, t_xss_aug, m_xss_aug, min_number)
    d_high_aug, t_high_aug, m_high_aug = randomly_keep_elements(d_high_aug, t_high_aug, m_high_aug, min_number)
    d_benign_aug, t_benign_aug, m_benign_aug = randomly_keep_elements(d_benign_aug, t_benign_aug, m_benign_aug, min_number)

    # test
    d_sql_aug_t, t_sql_aug_t, m_sql_aug_t = augment_timestamps((d_sql[2:], t_sql[2:], m_sql[2:]), n=3)
    d_command_aug_t, t_command_aug_t, m_command_aug_t = augment_timestamps((d_command[2:], t_command[2:], m_command[2:]), n=3)
    d_backdoor_aug_t, t_backdoor_aug_t, m_backdoor_aug_t = augment_timestamps((d_backdoor[2:], t_backdoor[2:], m_backdoor[2:]), n=3)
    d_uploading_aug_t, t_uploading_aug_t, m_uploading_aug_t = augment_timestamps((d_uploading[2:], t_uploading[2:], m_uploading[2:]), n=3)
    d_xss_aug_t, t_xss_aug_t, m_xss_aug_t = augment_timestamps((d_xss[2:], t_xss[2:], m_xss[2:]), n=3)
    d_high_aug_t, t_high_aug_t, m_high_aug_t = augment_timestamps((d_high[2:], t_high[2:], m_high[2:]), n=3)
    d_benign_aug_t, t_benign_aug_t, m_benign_aug_t = augment_timestamps((d_benign[2:], t_benign[2:], m_benign[2:]), n=3)
    
    # training, val, test split
    print("preparing training, val and test sets...")
    
    x_train = np.concatenate((d_sql_aug, d_command_aug, d_backdoor_aug, d_uploading_aug, d_xss_aug, d_high_aug, d_benign_aug), axis=0)
    t_train = np.concatenate((t_sql_aug, t_command_aug, t_backdoor_aug, t_uploading_aug, t_xss_aug, t_high_aug, t_benign_aug), axis=0)
    m_train = np.concatenate((m_sql_aug, m_command_aug, m_backdoor_aug, m_uploading_aug, m_xss_aug, m_high_aug, m_benign_aug), axis=0)
    y = d_sql_aug.shape[0]*[0] + d_command_aug.shape[0]*[1] + d_backdoor_aug.shape[0]*[2] + d_uploading_aug.shape[0]*[3] + d_xss_aug.shape[0]*[4] + d_high_aug.shape[0]*[5] + d_benign_aug.shape[0]*[6]
    y_train = np.array(y)

    # x_train, x_val, t_train, t_val, m_train, m_val, y_train, y_val = train_test_split(x_train, t_train, m_train, y_train, test_size=0.05, random_state=42)

    # x_test = np.concatenate((d_sql, d_command, d_backdoor, d_uploading, d_xss, d_high, d_benign), axis=0)
    # t_test = np.concatenate((t_sql, t_command, t_backdoor, t_uploading, t_xss, t_high, t_benign), axis=0)
    # m_test = np.concatenate((m_sql, m_command, m_backdoor, m_uploading, m_xss, m_high, m_benign), axis=0)
    # y = 3*[0] + 3*[1] + 3*[2] + 3*[3] + 3*[4] + 3*[5] + 3*[6]
    # y_test = np.array(y)
    
    x_test = np.concatenate((d_sql_aug_t, d_command_aug_t, d_backdoor_aug_t, d_uploading_aug_t, d_xss_aug_t, d_high_aug_t, d_benign_aug_t), axis=0)
    t_test = np.concatenate((t_sql_aug_t, t_command_aug_t, t_backdoor_aug_t, t_uploading_aug_t, t_xss_aug_t, t_high_aug_t, t_benign_aug_t), axis=0)
    m_test = np.concatenate((m_sql_aug_t, m_command_aug_t, m_backdoor_aug_t, m_uploading_aug_t, m_xss_aug_t, m_high_aug_t, m_benign_aug_t), axis=0)
    #y = [0, 1, 2, 3, 4, 5, 6]
    y = d_sql_aug_t.shape[0]*[0] + d_command_aug_t.shape[0]*[1] + d_backdoor_aug_t.shape[0]*[2] + d_uploading_aug_t.shape[0]*[3] + d_xss_aug_t.shape[0]*[4] + d_high_aug_t.shape[0]*[5] + d_benign_aug_t.shape[0]*[6]
    y_test = np.array(y)

    print(f"- Training samples: {x_train.shape[0]}")
    # print(f"- Validation samples: {x_val.shape[0]}")
    print(f"- Testing samples: {x_test.shape[0]}")

    # TF dataset
    BATCH_SIZE = 8
    BUFFER_SIZE = BATCH_SIZE * 2
    AUTO = tf.data.AUTOTUNE
    
    train_ds = tf.data.Dataset.from_tensor_slices(((x_train, t_train, m_train), y_train))
    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

    # val_ds = tf.data.Dataset.from_tensor_slices(((x_val, t_val, m_val), y_val))
    # val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTO)
    
    test_ds = tf.data.Dataset.from_tensor_slices(((x_test, t_test, m_test), y_test))
    test_ds = test_ds.batch(1).prefetch(AUTO)

    end_time = time.time()
    execution_time = int(end_time - start_time)
    print(f"dataset created succesfully in {execution_time} seconds!")
    
    return train_ds, test_ds, (x_test, t_test, m_test, y_test)

def prepare_dataset_v4(data, packet_length, max_num_packets, augmentations, augmentation_level):

    start_time = time.time()
    print("reading pcap files and creating flows...")
    
    flows_sql, flows_command, flows_backdoor, flows_uploading, flows_xss, flows_high, flows_benign = data
    
    # preprocess
    print("preprocessing flows...")
    
    data_sql = trim_or_pad(flows_sql, packet_length, max_num_packets)
    data_command = trim_or_pad(flows_command, packet_length, max_num_packets)
    data_backdoor = trim_or_pad(flows_backdoor, packet_length, max_num_packets)
    data_uploading = trim_or_pad(flows_uploading, packet_length, max_num_packets)
    data_xss = trim_or_pad(flows_xss, packet_length, max_num_packets)
    data_high = trim_or_pad(flows_high, packet_length, max_num_packets)
    data_benign = trim_or_pad(flows_benign, packet_length, max_num_packets)

    d_sql, t_sql, m_sql = create_masks(data_sql, max_num_packets)
    d_command, t_command, m_command = create_masks(data_command, max_num_packets)
    d_backdoor, t_backdoor, m_backdoor = create_masks(data_backdoor, max_num_packets)
    d_uploading, t_uploading, m_uploading = create_masks(data_uploading, max_num_packets)
    d_xss, t_xss, m_xss = create_masks(data_xss, max_num_packets)
    d_high, t_high, m_high = create_masks(data_high, max_num_packets)
    d_benign, t_benign, m_benign = create_masks(data_benign, max_num_packets)

    # augmentation
    print("augmenting flows...")
    if max_num_packets == 30:
        n = [9, 9, 9, 9, 9, 9, 9]
    elif max_num_packets == 50:
        n = [5, 5, 5, 9, 5, 5, 5]
    elif max_num_packets == 100:
        n = [2, 3, 3, 9, 5, 2, 2]
    elif max_num_packets == 200:
        n = [2, 3, 3, 9, 5, 1, 1]
    
    if augmentation_level == "mid":
        n = [x*2 for x  in n]
    if augmentation_level == "high":
        n = [x*4 for x  in n]

    augment_fn = augment_flows_v2
    # number of packets for each flow in class 0: 132, 331, 253
    d_sql_aug, t_sql_aug, m_sql_aug = augment_fn((d_sql[:2], t_sql[:2], m_sql[:2]), augmentations, max_num_packets, n=n[0], k=1)
    # number of packets for each flow in class 1: 75, 72, 86
    d_command_aug, t_command_aug, m_command_aug = augment_fn((d_command[:2], t_command[:2], m_command[:2]), augmentations, max_num_packets, n=n[1], k=1)
    # number of packets for each flow in class 2: 86, 94, 84
    d_backdoor_aug, t_backdoor_aug, m_backdoor_aug = augment_fn((d_backdoor[:2], t_backdoor[:2], m_backdoor[:2]), augmentations, max_num_packets, n=n[2], k=1)
    # number of packets for each flow in class 3: 28, 28, 29
    d_uploading_aug, t_uploading_aug, m_uploading_aug = augment_fn((d_uploading[:2], t_uploading[:2], m_uploading[:2]), augmentations, max_num_packets, n=n[3], k=1)
    # number of packets for each flow in class 4: 50, 55, 40
    d_xss_aug, t_xss_aug, m_xss_aug = augment_fn((d_xss[:2], t_xss[:2], m_xss[:2]), augmentations, max_num_packets, n=n[4], k=1)
    # number of packets for each flow in class 5: 5042, 1006, 1907
    d_high_aug, t_high_aug, m_high_aug = augment_fn((d_high[:2], t_high[:2], m_high[:2]), augmentations, max_num_packets, n=n[5], k=1)
    # number of packets for each flow in class 6: 198, 900, 206
    d_benign_aug, t_benign_aug, m_benign_aug = augment_fn((d_benign[:2], t_benign[:2], m_benign[:2]), augmentations, max_num_packets, n=n[6], k=1)
    
    min_number = np.min([d_sql_aug.shape[0], d_command_aug.shape[0], d_backdoor_aug.shape[0], d_uploading_aug.shape[0], d_xss_aug.shape[0], d_high_aug.shape[0], d_benign_aug.shape[0]])

    d_sql_aug, t_sql_aug, m_sql_aug = randomly_keep_elements(d_sql_aug, t_sql_aug, m_sql_aug, min_number)
    d_command_aug, t_command_aug, m_command_aug = randomly_keep_elements(d_command_aug, t_command_aug, m_command_aug, min_number)
    d_backdoor_aug, t_backdoor_aug, m_backdoor_aug = randomly_keep_elements(d_backdoor_aug, t_backdoor_aug, m_backdoor_aug, min_number)
    d_uploading_aug, t_uploading_aug, m_uploading_aug = randomly_keep_elements(d_uploading_aug, t_uploading_aug, m_uploading_aug, min_number)
    d_xss_aug, t_xss_aug, m_xss_aug = randomly_keep_elements(d_xss_aug, t_xss_aug, m_xss_aug, min_number)
    d_high_aug, t_high_aug, m_high_aug = randomly_keep_elements(d_high_aug, t_high_aug, m_high_aug, min_number)
    d_benign_aug, t_benign_aug, m_benign_aug = randomly_keep_elements(d_benign_aug, t_benign_aug, m_benign_aug, min_number)

    # test
    d_sql_aug_t, t_sql_aug_t, m_sql_aug_t = augment_timestamps((d_sql[2:], t_sql[2:], m_sql[2:]), n=3)
    d_command_aug_t, t_command_aug_t, m_command_aug_t = augment_timestamps((d_command[2:], t_command[2:], m_command[2:]), n=3)
    d_backdoor_aug_t, t_backdoor_aug_t, m_backdoor_aug_t = augment_timestamps((d_backdoor[2:], t_backdoor[2:], m_backdoor[2:]), n=3)
    d_uploading_aug_t, t_uploading_aug_t, m_uploading_aug_t = augment_timestamps((d_uploading[2:], t_uploading[2:], m_uploading[2:]), n=3)
    d_xss_aug_t, t_xss_aug_t, m_xss_aug_t = augment_timestamps((d_xss[2:], t_xss[2:], m_xss[2:]), n=3)
    d_high_aug_t, t_high_aug_t, m_high_aug_t = augment_timestamps((d_high[2:], t_high[2:], m_high[2:]), n=3)
    d_benign_aug_t, t_benign_aug_t, m_benign_aug_t = augment_timestamps((d_benign[2:], t_benign[2:], m_benign[2:]), n=3)
    
    # training, val, test split
    print("preparing training, val and test sets...")
    
    x_train = np.concatenate((d_sql_aug, d_command_aug, d_backdoor_aug, d_uploading_aug, d_xss_aug, d_high_aug, d_benign_aug), axis=0)
    t_train = np.concatenate((t_sql_aug, t_command_aug, t_backdoor_aug, t_uploading_aug, t_xss_aug, t_high_aug, t_benign_aug), axis=0)
    m_train = np.concatenate((m_sql_aug, m_command_aug, m_backdoor_aug, m_uploading_aug, m_xss_aug, m_high_aug, m_benign_aug), axis=0)
    y = d_sql_aug.shape[0]*[0] + d_command_aug.shape[0]*[1] + d_backdoor_aug.shape[0]*[2] + d_uploading_aug.shape[0]*[3] + d_xss_aug.shape[0]*[4] + d_high_aug.shape[0]*[5] + d_benign_aug.shape[0]*[6]
    y_train = np.array(y)

    # x_train, x_val, t_train, t_val, m_train, m_val, y_train, y_val = train_test_split(x_train, t_train, m_train, y_train, test_size=0.05, random_state=42)

    # x_test = np.concatenate((d_sql, d_command, d_backdoor, d_uploading, d_xss, d_high, d_benign), axis=0)
    # t_test = np.concatenate((t_sql, t_command, t_backdoor, t_uploading, t_xss, t_high, t_benign), axis=0)
    # m_test = np.concatenate((m_sql, m_command, m_backdoor, m_uploading, m_xss, m_high, m_benign), axis=0)
    # y = 3*[0] + 3*[1] + 3*[2] + 3*[3] + 3*[4] + 3*[5] + 3*[6]
    # y_test = np.array(y)
    
    x_test = np.concatenate((d_sql_aug_t, d_command_aug_t, d_backdoor_aug_t, d_uploading_aug_t, d_xss_aug_t, d_high_aug_t, d_benign_aug_t), axis=0)
    t_test = np.concatenate((t_sql_aug_t, t_command_aug_t, t_backdoor_aug_t, t_uploading_aug_t, t_xss_aug_t, t_high_aug_t, t_benign_aug_t), axis=0)
    m_test = np.concatenate((m_sql_aug_t, m_command_aug_t, m_backdoor_aug_t, m_uploading_aug_t, m_xss_aug_t, m_high_aug_t, m_benign_aug_t), axis=0)
    #y = [0, 1, 2, 3, 4, 5, 6]
    y = d_sql_aug_t.shape[0]*[0] + d_command_aug_t.shape[0]*[1] + d_backdoor_aug_t.shape[0]*[2] + d_uploading_aug_t.shape[0]*[3] + d_xss_aug_t.shape[0]*[4] + d_high_aug_t.shape[0]*[5] + d_benign_aug_t.shape[0]*[6]
    y_test = np.array(y)

    print(f"- Training samples: {x_train.shape[0]}")
    # print(f"- Validation samples: {x_val.shape[0]}")
    print(f"- Testing samples: {x_test.shape[0]}")

    # TF dataset
    BATCH_SIZE = 8
    BUFFER_SIZE = BATCH_SIZE * 2
    AUTO = tf.data.AUTOTUNE
    
    train_ds = tf.data.Dataset.from_tensor_slices(((x_train, t_train, m_train), y_train))
    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

    # val_ds = tf.data.Dataset.from_tensor_slices(((x_val, t_val, m_val), y_val))
    # val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTO)
    
    test_ds = tf.data.Dataset.from_tensor_slices(((x_test, t_test, m_test), y_test))
    test_ds = test_ds.batch(1).prefetch(AUTO)

    end_time = time.time()
    execution_time = int(end_time - start_time)
    print(f"dataset created succesfully in {execution_time} seconds!")
    
    return train_ds, test_ds, (x_test, t_test, m_test, y_test)

def prepare_dataset_v5(data, packet_length, max_num_packets, augmentations, augmentation_level):

    start_time = time.time()
    print("reading pcap files and creating flows...")
    
    flows_sql, flows_command, flows_backdoor, flows_uploading, flows_xss, flows_high, flows_benign = data
    
    # preprocess
    print("preprocessing flows...")
    
    data_sql = trim_or_pad(flows_sql, packet_length, max_num_packets)
    data_command = trim_or_pad(flows_command, packet_length, max_num_packets)
    data_backdoor = trim_or_pad(flows_backdoor, packet_length, max_num_packets)
    data_uploading = trim_or_pad(flows_uploading, packet_length, max_num_packets)
    data_xss = trim_or_pad(flows_xss, packet_length, max_num_packets)
    data_high = trim_or_pad(flows_high, packet_length, max_num_packets)
    data_benign = trim_or_pad(flows_benign, packet_length, max_num_packets)

    d_sql, t_sql, m_sql = create_masks(data_sql, max_num_packets)
    d_command, t_command, m_command = create_masks(data_command, max_num_packets)
    d_backdoor, t_backdoor, m_backdoor = create_masks(data_backdoor, max_num_packets)
    d_uploading, t_uploading, m_uploading = create_masks(data_uploading, max_num_packets)
    d_xss, t_xss, m_xss = create_masks(data_xss, max_num_packets)
    d_high, t_high, m_high = create_masks(data_high, max_num_packets)
    d_benign, t_benign, m_benign = create_masks(data_benign, max_num_packets)

    # augmentation
    print("augmenting flows...")
    if max_num_packets == 30:
        n = [9, 9, 9, 9, 9, 9, 9]
    elif max_num_packets == 50:
        n = [5, 5, 5, 9, 5, 5, 5]
    elif max_num_packets == 100:
        n = [2, 3, 3, 9, 5, 2, 2]
    elif max_num_packets == 200:
        n = [2, 3, 3, 9, 5, 1, 1]
    
    if augmentation_level == "mid":
        n = [x*2 for x  in n]
    if augmentation_level == "high":
        n = [x*4 for x  in n]

    augment_fn = augment_flows_v2
    # number of packets for each flow in class 0: 132, 331, 253
    d_sql_aug, t_sql_aug, m_sql_aug = augment_fn((d_sql[:2], t_sql[:2], m_sql[:2]), augmentations, max_num_packets, n=n[0], k=1)
    # number of packets for each flow in class 1: 75, 72, 86
    d_command_aug, t_command_aug, m_command_aug = augment_fn((d_command[:2], t_command[:2], m_command[:2]), augmentations, max_num_packets, n=n[1], k=1)
    # number of packets for each flow in class 2: 86, 94, 84
    d_backdoor_aug, t_backdoor_aug, m_backdoor_aug = augment_fn((d_backdoor[:2], t_backdoor[:2], m_backdoor[:2]), augmentations, max_num_packets, n=n[2], k=1)
    # number of packets for each flow in class 3: 28, 28, 29
    d_uploading_aug, t_uploading_aug, m_uploading_aug = augment_fn((d_uploading[:2], t_uploading[:2], m_uploading[:2]), augmentations, max_num_packets, n=n[3], k=1)
    # number of packets for each flow in class 4: 50, 55, 40
    d_xss_aug, t_xss_aug, m_xss_aug = augment_fn((d_xss[:2], t_xss[:2], m_xss[:2]), augmentations, max_num_packets, n=n[4], k=1)
    # number of packets for each flow in class 5: 5042, 1006, 1907
    d_high_aug, t_high_aug, m_high_aug = augment_fn((d_high[:2], t_high[:2], m_high[:2]), augmentations, max_num_packets, n=n[5], k=1)
    # number of packets for each flow in class 6: 198, 900, 206
    d_benign_aug, t_benign_aug, m_benign_aug = augment_fn((d_benign[:2], t_benign[:2], m_benign[:2]), augmentations, max_num_packets, n=n[6], k=1)
    
    min_number = np.min([d_sql_aug.shape[0], d_command_aug.shape[0], d_backdoor_aug.shape[0], d_uploading_aug.shape[0], d_xss_aug.shape[0], d_high_aug.shape[0], d_benign_aug.shape[0]])

    d_sql_aug, t_sql_aug, m_sql_aug = randomly_keep_elements(d_sql_aug, t_sql_aug, m_sql_aug, min_number)
    d_command_aug, t_command_aug, m_command_aug = randomly_keep_elements(d_command_aug, t_command_aug, m_command_aug, min_number)
    d_backdoor_aug, t_backdoor_aug, m_backdoor_aug = randomly_keep_elements(d_backdoor_aug, t_backdoor_aug, m_backdoor_aug, min_number)
    d_uploading_aug, t_uploading_aug, m_uploading_aug = randomly_keep_elements(d_uploading_aug, t_uploading_aug, m_uploading_aug, min_number)
    d_xss_aug, t_xss_aug, m_xss_aug = randomly_keep_elements(d_xss_aug, t_xss_aug, m_xss_aug, min_number)
    d_high_aug, t_high_aug, m_high_aug = randomly_keep_elements(d_high_aug, t_high_aug, m_high_aug, min_number)
    d_benign_aug, t_benign_aug, m_benign_aug = randomly_keep_elements(d_benign_aug, t_benign_aug, m_benign_aug, min_number)

    # test
    d_sql_aug_t, t_sql_aug_t, m_sql_aug_t = augment_timestamps((d_sql[2:], t_sql[2:], m_sql[2:]), n=3)
    d_command_aug_t, t_command_aug_t, m_command_aug_t = augment_timestamps((d_command[2:], t_command[2:], m_command[2:]), n=3)
    d_backdoor_aug_t, t_backdoor_aug_t, m_backdoor_aug_t = augment_timestamps((d_backdoor[2:], t_backdoor[2:], m_backdoor[2:]), n=3)
    d_uploading_aug_t, t_uploading_aug_t, m_uploading_aug_t = augment_timestamps((d_uploading[2:], t_uploading[2:], m_uploading[2:]), n=3)
    d_xss_aug_t, t_xss_aug_t, m_xss_aug_t = augment_timestamps((d_xss[2:], t_xss[2:], m_xss[2:]), n=3)
    d_high_aug_t, t_high_aug_t, m_high_aug_t = augment_timestamps((d_high[2:], t_high[2:], m_high[2:]), n=3)
    d_benign_aug_t, t_benign_aug_t, m_benign_aug_t = augment_timestamps((d_benign[2:], t_benign[2:], m_benign[2:]), n=3)
    
    # training, val, test split
    print("preparing training, val and test sets...")
    
    x_train = np.concatenate((d_sql_aug, d_command_aug, d_backdoor_aug, d_uploading_aug, d_xss_aug, d_high_aug, d_benign_aug), axis=0)
    t_train = np.concatenate((t_sql_aug, t_command_aug, t_backdoor_aug, t_uploading_aug, t_xss_aug, t_high_aug, t_benign_aug), axis=0)
    m_train = np.concatenate((m_sql_aug, m_command_aug, m_backdoor_aug, m_uploading_aug, m_xss_aug, m_high_aug, m_benign_aug), axis=0)
    y = d_sql_aug.shape[0]*[0] + d_command_aug.shape[0]*[1] + d_backdoor_aug.shape[0]*[2] + d_uploading_aug.shape[0]*[3] + d_xss_aug.shape[0]*[4] + d_high_aug.shape[0]*[5] + d_benign_aug.shape[0]*[6]
    y_train = np.array(y)

    x_train, x_val, t_train, t_val, m_train, m_val, y_train, y_val = train_test_split(x_train, t_train, m_train, y_train, test_size=0.05, random_state=42)

    # x_test = np.concatenate((d_sql, d_command, d_backdoor, d_uploading, d_xss, d_high, d_benign), axis=0)
    # t_test = np.concatenate((t_sql, t_command, t_backdoor, t_uploading, t_xss, t_high, t_benign), axis=0)
    # m_test = np.concatenate((m_sql, m_command, m_backdoor, m_uploading, m_xss, m_high, m_benign), axis=0)
    # y = 3*[0] + 3*[1] + 3*[2] + 3*[3] + 3*[4] + 3*[5] + 3*[6]
    # y_test = np.array(y)
    
    x_test = np.concatenate((d_sql_aug_t, d_command_aug_t, d_backdoor_aug_t, d_uploading_aug_t, d_xss_aug_t, d_high_aug_t, d_benign_aug_t), axis=0)
    t_test = np.concatenate((t_sql_aug_t, t_command_aug_t, t_backdoor_aug_t, t_uploading_aug_t, t_xss_aug_t, t_high_aug_t, t_benign_aug_t), axis=0)
    m_test = np.concatenate((m_sql_aug_t, m_command_aug_t, m_backdoor_aug_t, m_uploading_aug_t, m_xss_aug_t, m_high_aug_t, m_benign_aug_t), axis=0)
    y = d_sql_aug_t.shape[0]*[0] + d_command_aug_t.shape[0]*[1] + d_backdoor_aug_t.shape[0]*[2] + d_uploading_aug_t.shape[0]*[3] + d_xss_aug_t.shape[0]*[4] + d_high_aug_t.shape[0]*[5] + d_benign_aug_t.shape[0]*[6]
    y_test = np.array(y)

    print(f"- Training samples: {x_train.shape[0]}")
    print(f"- Validation samples: {x_val.shape[0]}")
    print(f"- Testing samples: {x_test.shape[0]}")

    # TF dataset
    BATCH_SIZE = 8
    BUFFER_SIZE = BATCH_SIZE * 2
    AUTO = tf.data.AUTOTUNE
    
    train_ds = tf.data.Dataset.from_tensor_slices(((x_train, t_train, m_train), y_train))
    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

    val_ds = tf.data.Dataset.from_tensor_slices(((x_val, t_val, m_val), y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTO)
    
    #test_ds = tf.data.Dataset.from_tensor_slices(((x_test, t_test, m_test), y_test))
    #test_ds = test_ds.batch(1).prefetch(AUTO)

    end_time = time.time()
    execution_time = int(end_time - start_time)
    print(f"dataset created succesfully in {execution_time} seconds!")
    
    return train_ds, val_ds, (x_test, t_test, m_test, y_test)

#class TransformerEncoder(tf.keras.layers.Layer):
#    def __init__(self, d_model, num_heads, dff, seq_len, encoding, activation='gelu', norm='layer', dropout_rate=0.1):
#        super().__init__()
#        self.d_model = d_model
#        self.encoding = encoding
#        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
#        #self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
#        self.ffn = tf.keras.Sequential([
#            Dense(dff, activation),
#            Dense(d_model)
#        ])
#        if norm == 'layer':
#            self.norm1 = LayerNormalization(epsilon=1e-6)
#            self.norm2 = LayerNormalization(epsilon=1e-6)
#        else:
#            self.norm1 = BatchNormalization()
#            self.norm2 = BatchNormalization()
#        self.dropout1 = Dropout(dropout_rate)
#        self.dropout2 = Dropout(dropout_rate)
#
#    def apply_rope(self, queries, keys, positions, d_model):
#        
#        # Step 1: Compute the sine and cosine components for RoPE
#        inv_freq = 1.0 / (10000 ** (tf.range(0, d_model, 2, dtype=tf.float32) / tf.cast(d_model, tf.float32)))
#        sin_pos = tf.sin(positions * inv_freq)  # Shape: (batch_size, seq_len, d_model/2)
#        cos_pos = tf.cos(positions * inv_freq)
#    
#        # Expand sin and cos for proper broadcasting
#        sin_pos = tf.concat([sin_pos, sin_pos], axis=-1)  # Duplicate for (d_model,)
#        cos_pos = tf.concat([cos_pos, cos_pos], axis=-1)
#    
#        # Step 2: Apply sine and cosine embeddings to query and key
#        query_rotated = queries * cos_pos + self.rotate_half(queries) * sin_pos
#        key_rotated = keys * cos_pos + self.rotate_half(keys) * sin_pos
#    
#        return query_rotated, key_rotated
#
#    def rotate_half(self, x):
#        d_model_half = tf.shape(x)[-1] // 2
#        x1, x2 = x[..., :d_model_half], x[..., d_model_half:]  # Split the last dimension
#        return tf.concat([-x2, x1], axis=-1)
#        
#    def compute_output_shape(self, input_shape):
#        return input_shape
#        
#    def call(self, x, mask, t=None, training=False):
#        
#        seq_len = tf.shape(x)[1]
#        
#        mask = tf.expand_dims(mask, axis=1)  # Add dimension for heads
#        mask = tf.expand_dims(mask, axis=1)  # Add dimension for queries
#        
#        if self.encoding == "rope":
#            positions = tf.range(seq_len, dtype=tf.float32)[tf.newaxis, :, tf.newaxis]
#            query, key = self.apply_rope(x, x, positions, self.d_model)
#            attn_output = self.mha(query, x, key, attention_mask=mask)
#        elif self.encoding == "dyn_rope":
#            positions = tf.expand_dims(t, -1)
#            query, key = self.apply_rope(x, x, positions, self.d_model)
#            attn_output = self.mha(query, x, key, attention_mask=mask)
#        else:
#            attn_output = self.mha(x, x, x, attention_mask=mask)
#        
#        attn_output = self.dropout1(attn_output, training=training)
#        out1 = self.norm1(x + attn_output, training=training)
#        
#        ffn_output = self.ffn(out1)
#        ffn_output = self.dropout2(ffn_output, training=training)
#        out2 = self.norm2(out1 + ffn_output, training=training)
#        
#        return out2


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, seq_len, encoding, activation='gelu', norm='layer', dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.encoding = encoding
        if encoding == "regl":
            self.mha = RelativeGlobalAttention(d_model, num_heads, d_head=d_model, max_len=seq_len, dropout=0.1)
        else:
            self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation),
            Dense(d_model)
        ])
        if norm == 'layer':
            self.norm1 = LayerNormalization(epsilon=1e-6)
            self.norm2 = LayerNormalization(epsilon=1e-6)
        else:
            self.norm1 = BatchNormalization()
            self.norm2 = BatchNormalization()
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def apply_rope(self, queries, keys, positions, d_model):
        
        # Step 1: Compute the sine and cosine components for RoPE
        inv_freq = 1.0 / (10000 ** (tf.range(0, d_model, 2, dtype=tf.float32) / tf.cast(d_model, tf.float32)))
        sin_pos = tf.sin(positions * inv_freq)  # Shape: (batch_size, seq_len, d_model/2)
        cos_pos = tf.cos(positions * inv_freq)
    
        # Expand sin and cos for proper broadcasting
        sin_pos = tf.concat([sin_pos, sin_pos], axis=-1)  # Duplicate for (d_model,)
        cos_pos = tf.concat([cos_pos, cos_pos], axis=-1)
    
        # Step 2: Apply sine and cosine embeddings to query and key
        query_rotated = queries * cos_pos + self.rotate_half(queries) * sin_pos
        key_rotated = keys * cos_pos + self.rotate_half(keys) * sin_pos
    
        return query_rotated, key_rotated

    def rotate_half(self, x):
        d_model_half = tf.shape(x)[-1] // 2
        x1, x2 = x[..., :d_model_half], x[..., d_model_half:]  # Split the last dimension
        return tf.concat([-x2, x1], axis=-1)
        
    def compute_output_shape(self, input_shape):
        return input_shape
        
    def call(self, x, mask, t=None, training=False):
        
        seq_len = tf.shape(x)[1]
        
        mask = tf.expand_dims(mask, axis=1)  # Add dimension for heads
        mask = tf.expand_dims(mask, axis=1)  # Add dimension for queries
        
        if self.encoding == "rope":
            positions = tf.range(seq_len, dtype=tf.float32)[tf.newaxis, :, tf.newaxis]
            query, key = self.apply_rope(x, x, positions, self.d_model)
            attn_output = self.mha(query, x, key, attention_mask=mask)
        elif self.encoding == "dyn_rope":
            positions = tf.expand_dims(t, -1)
            query, key = self.apply_rope(x, x, positions, self.d_model)
            attn_output = self.mha(query, x, key, attention_mask=mask)
        elif self.encoding == "regl":
            attn_output = self.mha(x, attention_mask=mask)
        else:
            attn_output = self.mha(x, x, x, attention_mask=mask)
        
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output, training=training)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output, training=training)
        
        return out2

    def build(self, input_shape):
        """Ensures all layers are properly initialized and registered."""
        batch_size, seq_len, d_model = input_shape
    
        if self.encoding == "regl":
            self.mha.build((batch_size, seq_len, d_model))
        else:
            self.mha.build((batch_size, seq_len, d_model), (batch_size, seq_len, d_model))
    
        self.ffn.build((batch_size, seq_len, d_model))
        self.norm1.build((batch_size, seq_len, d_model))
        self.norm2.build((batch_size, seq_len, d_model))
        self.dropout1.build((batch_size, seq_len, d_model))
        self.dropout2.build((batch_size, seq_len, d_model))
    
        super(TransformerEncoder, self).build(input_shape)

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.mha.num_heads,
            "dff": self.ffn.layers[0].units,  # Extract feedforward dimension
            "seq_len": self.max_len if hasattr(self, "max_len") else None,
            "encoding": self.encoding,
            "activation": self.ffn.layers[0].activation.__name__,  # Extract activation function
            "norm": "layer" if isinstance(self.norm1, LayerNormalization) else "batch",
            "dropout_rate": self.dropout1.rate,
        })
        return config




class RelativeGlobalAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_head, max_len=1024, dropout=0.1):
        super(RelativeGlobalAttention, self).__init__()
        self.max_len = max_len
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_model = d_model
        self.proj_dim = num_heads * d_head # Projection dimension
        
        # Additional projection layers for query, key, and value if d_model != num_heads * d_head
        if self.proj_dim != d_model:
            self.out_proj = tf.keras.layers.Dense(self.d_model, use_bias=True)  # Final projection back to d_model
        else:
            self.out_proj = None

        self.q_proj = tf.keras.layers.Dense(self.proj_dim, use_bias=False)
        self.k_proj = tf.keras.layers.Dense(self.proj_dim, use_bias=False)
        self.v_proj = tf.keras.layers.Dense(self.proj_dim, use_bias=False)

        self.dropout = tf.keras.layers.Dropout(dropout)
        
        self.Er = self.add_weight(name="Er", shape=(max_len, d_head), initializer=tf.keras.initializers.RandomNormal(), trainable=True)
        
    def call(self, x, attention_mask=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Project input for q, k, v separately if needed
        q_input = self.q_proj(x)
        k_input = self.k_proj(x)
        v_input = self.v_proj(x)
        
        k_t = tf.transpose(tf.reshape(k_input, (batch_size, seq_len, self.num_heads, self.d_head)), perm=[0, 2, 3, 1])
        v = tf.transpose(tf.reshape(v_input, (batch_size, seq_len, self.num_heads, self.d_head)), perm=[0, 2, 1, 3])
        q = tf.transpose(tf.reshape(q_input, (batch_size, seq_len, self.num_heads, self.d_head)), perm=[0, 2, 1, 3])
        
        start = self.max_len - seq_len
        Er_t = tf.transpose(self.Er[start:, :])
        QEr = tf.matmul(q, Er_t)
        Srel = self.skew(QEr)
        
        QK_t = tf.matmul(q, k_t)
        attn = (QK_t + Srel) / tf.math.sqrt(tf.cast(self.d_head, tf.float32))

        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, dtype=tf.float32)
            attention_mask = tf.reshape(attention_mask, (batch_size, 1, 1, seq_len))
            attn += (1.0 - attention_mask) * -1e4 #9  # Apply mask before softmax
        
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, seq_len, self.proj_dim))
        
        if self.out_proj is not None:
            out = self.out_proj(out)  # Project back to d_model
        
        return self.dropout(out)
    
    def skew(self, QEr):
        padded = tf.pad(QEr, [[0, 0], [0, 0], [0, 0], [1, 0]])
        reshaped = tf.reshape(padded, (tf.shape(padded)[0], tf.shape(padded)[1], tf.shape(padded)[3], tf.shape(padded)[2]))
        Srel = reshaped[:, :, 1:, :]
        return Srel

    def build(self, input_shape):
        """ Ensures Dense layers are properly built and registered in model.summary(). """
        if self.q_proj is not None:
            self.q_proj.build(input_shape)
            self.k_proj.build(input_shape)
            self.v_proj.build(input_shape)
        
        #self.key.build((input_shape[0], input_shape[1], self.proj_dim))
        #self.value.build((input_shape[0], input_shape[1], self.proj_dim))
        #self.query.build((input_shape[0], input_shape[1], self.proj_dim))
        
        if self.out_proj is not None:
            self.out_proj.build((input_shape[0], input_shape[1], self.proj_dim))

        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        
        super(RelativeGlobalAttention, self).build(input_shape)
        
    def get_config(self):
        config = super(RelativeGlobalAttention, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_head": self.d_head,
            "max_len": self.max_len,
            "dropout": float(self.dropout.rate),  # Ensure dropout is serializable
        })
        return config


class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_position = max_position
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        positions = tf.range(self.max_position, dtype=tf.float32)[:, tf.newaxis]
        indices = tf.range(self.embedding_dim, dtype=tf.float32)[tf.newaxis, :]
        div_term = tf.pow(10000.0, (2 * (indices // 2)) / tf.cast(self.embedding_dim, tf.float32))
        
        pos_enc = tf.where(tf.cast(indices, tf.int32) % 2 == 0,
                           tf.sin(positions / div_term),  # Even indices -> sin
                           tf.cos(positions / div_term))  # Odd indices -> cos
        
        #self.positional_encoding = tf.Variable(pos_enc, trainable=False, name="positional_encoding")
        pos_enc = pos_enc.numpy()
        self.positional_encoding = self.add_weight(
            name="positional_encoding",
            shape=pos_enc.shape,
            initializer=tf.constant_initializer(pos_enc),
            trainable=False
        )
        super().build(input_shape)

    def call(self, x):
        seq_length = tf.shape(x)[1]
        pos_enc = self.positional_encoding[:seq_length, :]
        return pos_enc

class DynamicSinusoidalPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def call(self, positions, mask):
        positions = tf.expand_dims(positions, -1) # Shape: (batch_size, sequence_length, 1)
        indices = tf.range(self.embedding_dim, dtype=tf.float32)
        indices = tf.expand_dims(indices, 0)
        indices = tf.expand_dims(indices, 0) # Shape: (1, 1, embedding_dim)
        div_term = tf.pow(10000.0, (2 * (indices // 2)) / tf.cast(self.embedding_dim, tf.float32))
        
        pos_enc = tf.where(tf.cast(indices, tf.int32) % 2 == 0,
                           tf.sin(positions / div_term),  # Even indices -> sin
                           tf.cos(positions / div_term))  # Odd indices -> cos
        
        
        seq_length = tf.shape(mask)[1]
        pos_enc = pos_enc[:, :seq_length, :]
        return pos_enc

class FourierPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.frequencies = self.add_weight(shape=(d_model // 2,), initializer="random_normal", trainable=True)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(seq_len, dtype=tf.float32)[:, None]
        angles = positions * self.frequencies[None, :]
        encoding = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)
        return encoding

class DynamicFourierPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.frequencies = self.add_weight(shape=(d_model // 2,), initializer="random_normal", trainable=True)

    def call(self, x, t):
        seq_len = tf.shape(x)[1]
        positions = tf.expand_dims(t, -1)
        angles = positions * self.frequencies[None, :]
        encoding = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)
        return encoding

class EmbeddingPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super(EmbeddingPositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.positional_embedding = tf.keras.layers.Embedding(
            input_dim=max_len, 
            output_dim=d_model
        )

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)  # Generate positions [0, 1, ..., seq_len-1]
        position_embeddings = self.positional_embedding(positions)
        return position_embeddings

class ConvPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.conv = tf.keras.layers.Conv1D(filters=d_model, kernel_size=kernel_size, padding="same")
        
    def call(self, x):
        pos_enc = self.conv(x)
        return pos_enc

class ClassificationEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, **kwargs):
        super(ClassificationEmbeddingLayer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        # Create a trainable classification embedding
        self.classification_embedding = self.add_weight(
            shape=(1, 1, self.embedding_dim),
            initializer="random_normal",
            trainable=True,
            name="classification_embedding"
        )
        super(ClassificationEmbeddingLayer, self).build(input_shape)

    def call(self, inputs):
        # Repeat the classification embedding for the batch size
        batch_size = tf.shape(inputs)[0]
        classification_embedding_repeated = tf.tile(self.classification_embedding, [batch_size, 1, 1])

        # Remove the last token from the input sequence
        inputs_trimmed = inputs[:, :-1, :]  # Shape: (batch_size, seq_len - 1, embedding_dim)

        # Concatenate the classification embedding with the input embeddings
        return tf.concat([classification_embedding_repeated, inputs_trimmed], axis=1)

    def compute_output_shape(self, input_shape):
        # Add 1 to the sequence length for the classification embedding
        return input_shape

def create_transformer_model(packet_length, num_classes, max_len, activation='gelu', norm='batch', encoding='sin', d_model=128, num_heads=4, dff=512, num_layers=3, cls_emb=False, dropout_rate=0.1):
    
    input_layer = Input(shape=(None, packet_length), name='flows')
    mask_layer = Input(shape=(None,), name='masks')
    if encoding in ["dyn_sin", "dyn_fourier", "dyn_rope"]:
        time_layer = Input(shape=(None,), name='timestamps')

    x = Dense(d_model)(input_layer)  # Linear Embedding layer
    
    if cls_emb:
        x = ClassificationEmbeddingLayer(d_model)(x)
    
    if encoding == 'sin':
        e = SinusoidalPositionalEncoding(max_position=max_len, embedding_dim=d_model)(mask_layer)
        x = x + e
    if encoding == 'dyn_sin':
        e = DynamicSinusoidalPositionalEncoding(embedding_dim=d_model)(time_layer, mask_layer)
        x = x + e
    if encoding == 'fourier':
        e = FourierPositionalEncoding(d_model)(input_layer)
        x = x + e
    if encoding == 'dyn_fourier':
        e = DynamicFourierPositionalEncoding(d_model)(input_layer, time_layer)
        x = x + e
    if encoding == 'embedding':
        e = EmbeddingPositionalEncoding(max_len, d_model)(input_layer)
        x = x + e
    if encoding == 'conv':
        e = ConvPositionalEncoding(d_model)(input_layer)
        x = x + e
    
    # Stack Transformer Encoder Layers
    for _ in range(num_layers):
        if encoding in ["dyn_sin", "dyn_fourier", "dyn_rope"]:
            x = TransformerEncoder(d_model, num_heads, dff, max_len, encoding, activation, norm, dropout_rate)(x, mask_layer, time_layer)
        else:
            x = TransformerEncoder(d_model, num_heads, dff, max_len, encoding, activation, norm, dropout_rate)(x, mask_layer)

    if cls_emb:
        # take first token
        x = x[:, 0, :]
    else:
        # Global average pooling to reduce sequence dimension
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    if encoding in ["dyn_sin", "dyn_fourier", "dyn_rope"]:
        model = Model(inputs=[input_layer, time_layer, mask_layer], outputs=output_layer)
    else:
        model = Model(inputs=[input_layer, mask_layer], outputs=output_layer)
    
    return model

class TextLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file):
        super(TextLoggerCallback, self).__init__()
        self.log_file = log_file
        with open(log_file, 'w') as file:
            pass

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.log_file, "a") as f:
            log_entry = f"Epoch {epoch + 1}: " + ", ".join([f"{k}={v:.10f}" for k, v in logs.items()]) + "\n"
            f.write(log_entry)

def custom_loss(y_true, y_pred, masks):

    lengths = []
    for mask in masks:
        lengths.append(int(np.sum(mask)))
    
    base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)

    # original
    # weights = 1.0 / tf.cast(lengths, dtype=tf.float32)
    
    # 1st option
    # weights = weights / 3.994987130920391
    
    # 2nd
    # weights = [mapping[x] for x in weights]

    # 3rd
    k = 0.1
    weights = [np.exp(-k*l) for l in lengths]

    # 4th
    # weights = [1 if x > 10 else 1.5 for x in lengths]

    # 5th
    # k = 0.1
    # weights = [np.exp(-k*l)+1 for l in lengths]
    
    return tf.reduce_mean(base_loss * weights)

class Found:
    def __init__(self, x_test, t_test, m_test, y_test, num_classes):
        self.x_test = x_test
        self.t_test = t_test
        self.m_test = m_test
        self.y_test = y_test
        self.num_classes = num_classes

    def on_epoch_end(self, epoch, model):

        pred_50 = [-1]*self.num_classes
        pred_80 = [-1]*self.num_classes
        pred_90 = [-1]*self.num_classes
        pred_95 = [-1]*self.num_classes
        pred_99 = [-1]*self.num_classes
        e_50 = [-1]*self.num_classes
        e_80 = [-1]*self.num_classes
        e_90 = [-1]*self.num_classes
        e_95 = [-1]*self.num_classes
        e_99 = [-1]*self.num_classes
        print_flag = [True]*self.num_classes
        
        cur_class = -1
        num_samples = self.x_test.shape[0]
        
        for x, t, m, y in zip(self.x_test, self.t_test, self.m_test, self.y_test):
            
            flag_50 = flag_80 = flag_90 = flag_95 = flag_99 = True
            
            if y != cur_class:
                cur_class = y
            
            counter = int(np.sum(m))
            
            augmented_x = [x[:k] for k in range(1, counter+1)]
            augmented_t = [t[:k] for k in range(1, counter+1)]
            augmented_m = [m[:k] for k in range(1, counter+1)]
            
            c = 0
            for x_, t_, m_ in zip(augmented_x, augmented_t, augmented_m):
                
                c = c + 1
                
                x_ = tf.expand_dims(x_, axis=0)
                t_ = tf.expand_dims(t_, axis=0)
                m_ = tf.expand_dims(m_, axis=0)
                
                pred = model.predict((x_, t_, m_), verbose=0)
                pred = tf.squeeze(pred)
                
                top_conf = tf.math.reduce_max(pred)
                predicted_class = tf.argmax(pred)
                
                if predicted_class == y and print_flag[y]:
                    print(f"Class {y} - Packet #{c}: {top_conf:.6f}")
                    print_flag[y] = False
                
                if flag_50 and top_conf >= 0.5:
                    pred_50[y] = predicted_class.numpy()
                    flag_50 = False
                    if predicted_class == y:
                        e_50[y] = c
                
                if flag_80 and top_conf >= 0.8:
                    pred_80[y] = predicted_class.numpy()
                    flag_80 = False
                    if predicted_class == y:
                        e_80[y] = c
                
                if flag_90 and top_conf >= 0.9:
                    pred_90[y] = predicted_class.numpy()
                    flag_90 = False
                    if predicted_class == y:
                        e_90[y] = c
                
                if flag_95 and top_conf >= 0.95:
                    pred_95[y] = predicted_class.numpy()
                    flag_95 = False
                    if predicted_class == y:
                        e_95[y] = c
                            
                if flag_99 and top_conf >= 0.99:
                    pred_99[y] = predicted_class.numpy()
                    flag_99 = False
                    if predicted_class == y:
                        e_99[y] = c
                            
                if not flag_50 and not flag_80 and not flag_90 and not flag_95 and not flag_99:
                    break
            
            if flag_50:
                pred_50[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_50[y] = c
            if flag_80:
                pred_80[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_80[y] = c
            if flag_90:
                pred_90[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_90[y] = c
            if flag_95:
                pred_95[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_95[y] = c
            if flag_99:
                pred_99[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_99[y] = c
        
        true_labels = list(range(self.num_classes))
        
        if int(np.max(e_50)) > -1:
            print("Confidence threshold = 50%")
            print(f"-Predictions: {pred_50}")
            print(f"-Earliness: {e_50}")
            #print(f"-Avg earliness: {np.mean(e_50):.3f}")
            print(f"-Max earliness: {int(np.max(e_50))}")
            correct_predictions = sum([true == pred for true, pred in zip(true_labels, pred_50)])
            print(f"-Accuracy: {correct_predictions}/{self.num_classes}")
            
        if int(np.max(e_80)) > -1:
            print("Confidence threshold = 80%")
            print(f"-Predictions: {pred_80}")
            print(f"-Earliness: {e_80}")
            #print(f"-Avg earliness: {np.mean(e_80):.3f}")
            print(f"-Max earliness: {int(np.max(e_80))}")
            correct_predictions = sum([true == pred for true, pred in zip(true_labels, pred_80)])
            print(f"-Accuracy: {correct_predictions}/{self.num_classes}")
        
        if int(np.max(e_90)) > -1:
            print("Confidence threshold = 90%")
            print(f"-Predictions: {pred_90}")
            print(f"-Earliness: {e_90}")
            #print(f"-Avg earliness: {np.mean(e_90):.3f}")
            print(f"-Max earliness: {int(np.max(e_90))}")
            correct_predictions = sum([true == pred for true, pred in zip(true_labels, pred_90)])
            print(f"-Accuracy: {correct_predictions}/{self.num_classes}")
        
        if int(np.max(e_95)) > -1:
            print("Confidence threshold = 95%")
            print(f"-Predictions: {pred_95}")
            print(f"-Earliness: {e_95}")
            #print(f"-Avg earliness: {np.mean(e_95):.3f}")
            print(f"-Max earliness: {int(np.max(e_95))}")
            correct_predictions = sum([true == pred for true, pred in zip(true_labels, pred_95)])
            print(f"-Accuracy: {correct_predictions}/{self.num_classes}")
        
        if int(np.max(e_99)) > -1:
            print("Confidence threshold = 99%")
            print(f"-Predictions: {pred_99}")
            print(f"-Earliness: {e_99}")
            #print(f"-Avg earliness: {np.mean(e_99):.3f}")
            print(f"-Max earliness: {int(np.max(e_99))}")
            correct_predictions = sum([true == pred for true, pred in zip(true_labels, pred_99)])
            print(f"-Accuracy: {correct_predictions}/{self.num_classes}") 
            
        return
        
    def on_training_end(self, model):
        
        pred_20 = [-1]*self.num_classes
        pred_25 = [-1]*self.num_classes
        pred_30 = [-1]*self.num_classes
        pred_35 = [-1]*self.num_classes
        pred_40 = [-1]*self.num_classes
        pred_45 = [-1]*self.num_classes
        pred_50 = [-1]*self.num_classes
        pred_55 = [-1]*self.num_classes
        pred_60 = [-1]*self.num_classes
        pred_65 = [-1]*self.num_classes
        pred_70 = [-1]*self.num_classes
        pred_75 = [-1]*self.num_classes
        pred_80 = [-1]*self.num_classes
        pred_85 = [-1]*self.num_classes
        pred_90 = [-1]*self.num_classes
        pred_95 = [-1]*self.num_classes
        pred_99 = [-1]*self.num_classes
        
        e_20 = [-1]*self.num_classes
        e_25 = [-1]*self.num_classes
        e_30 = [-1]*self.num_classes
        e_35 = [-1]*self.num_classes
        e_40 = [-1]*self.num_classes
        e_45 = [-1]*self.num_classes
        e_50 = [-1]*self.num_classes
        e_55 = [-1]*self.num_classes
        e_60 = [-1]*self.num_classes
        e_65 = [-1]*self.num_classes
        e_70 = [-1]*self.num_classes
        e_75 = [-1]*self.num_classes
        e_80 = [-1]*self.num_classes
        e_85 = [-1]*self.num_classes
        e_90 = [-1]*self.num_classes
        e_95 = [-1]*self.num_classes
        e_99 = [-1]*self.num_classes
        print_flag = [True]*self.num_classes
        
        cur_class = -1
        num_samples = self.x_test.shape[0]
        
        for x, t, m, y in zip(self.x_test, self.t_test, self.m_test, self.y_test):
            
            flag_20 = flag_25 = flag_30 = flag_35 = flag_40 = flag_45 = flag_50 = flag_55 = True
            flag_60 = flag_65 = flag_70 = flag_75 = flag_80 = flag_85 = flag_90 = flag_95 = flag_99 = True
            
            if y != cur_class:
                cur_class = y
            
            counter = int(np.sum(m))
            
            augmented_x = [x[:k] for k in range(1, counter+1)]
            augmented_t = [t[:k] for k in range(1, counter+1)]
            augmented_m = [m[:k] for k in range(1, counter+1)]
            
            c = 0
            for x_, t_, m_ in zip(augmented_x, augmented_t, augmented_m):
                
                c = c + 1
                
                x_ = tf.expand_dims(x_, axis=0)
                t_ = tf.expand_dims(t_, axis=0)
                m_ = tf.expand_dims(m_, axis=0)
                
                pred = model.predict((x_, t_, m_), verbose=0)
                pred = tf.squeeze(pred)
                
                top_conf = tf.math.reduce_max(pred)
                predicted_class = tf.argmax(pred)

                if flag_20 and top_conf >= 0.2:
                    pred_20[y] = predicted_class.numpy()
                    flag_20 = False
                    if predicted_class == y:
                        e_20[y] = c
                
                if flag_25 and top_conf >= 0.25:
                    pred_25[y] = predicted_class.numpy()
                    flag_25 = False
                    if predicted_class == y:
                        e_25[y] = c
                
                if flag_30 and top_conf >= 0.3:
                    pred_30[y] = predicted_class.numpy()
                    flag_30 = False
                    if predicted_class == y:
                        e_30[y] = c
                
                if flag_35 and top_conf >= 0.35:
                    pred_35[y] = predicted_class.numpy()
                    flag_35 = False
                    if predicted_class == y:
                        e_35[y] = c
                
                if flag_40 and top_conf >= 0.4:
                    pred_40[y] = predicted_class.numpy()
                    flag_40 = False
                    if predicted_class == y:
                        e_40[y] = c

                if flag_45 and top_conf >= 0.45:
                    pred_45[y] = predicted_class.numpy()
                    flag_45 = False
                    if predicted_class == y:
                        e_45[y] = c
                
                if flag_50 and top_conf >= 0.5:
                    pred_50[y] = predicted_class.numpy()
                    flag_50 = False
                    if predicted_class == y:
                        e_50[y] = c

                if flag_55 and top_conf >= 0.55:
                    pred_55[y] = predicted_class.numpy()
                    flag_55 = False
                    if predicted_class == y:
                        e_55[y] = c
                
                if flag_60 and top_conf >= 0.6:
                    pred_60[y] = predicted_class.numpy()
                    flag_60 = False
                    if predicted_class == y:
                        e_60[y] = c

                if flag_65 and top_conf >= 0.65:
                    pred_65[y] = predicted_class.numpy()
                    flag_65 = False
                    if predicted_class == y:
                        e_65[y] = c
                
                if flag_70 and top_conf >= 0.7:
                    pred_70[y] = predicted_class.numpy()
                    flag_70 = False
                    if predicted_class == y:
                        e_70[y] = c
                
                if flag_75 and top_conf >= 0.75:
                    pred_75[y] = predicted_class.numpy()
                    flag_75 = False
                    if predicted_class == y:
                        e_75[y] = c

                if flag_80 and top_conf >= 0.8:
                    pred_80[y] = predicted_class.numpy()
                    flag_80 = False
                    if predicted_class == y:
                        e_80[y] = c

                if flag_85 and top_conf >= 0.85:
                    pred_85[y] = predicted_class.numpy()
                    flag_85 = False
                    if predicted_class == y:
                        e_85[y] = c
                
                if flag_90 and top_conf >= 0.9:
                    pred_90[y] = predicted_class.numpy()
                    flag_90 = False
                    if predicted_class == y:
                        e_90[y] = c
                
                if flag_95 and top_conf >= 0.95:
                    pred_95[y] = predicted_class.numpy()
                    flag_95 = False
                    if predicted_class == y:
                        e_95[y] = c
                            
                if flag_99 and top_conf >= 0.99:
                    pred_99[y] = predicted_class.numpy()
                    flag_99 = False
                    if predicted_class == y:
                        e_99[y] = c
                
            if flag_20:
                pred_20[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_20[y] = c
            if flag_25:
                pred_25[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_25[y] = c
            if flag_30:
                pred_30[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_30[y] = c
            if flag_35:
                pred_35[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_35[y] = c
            if flag_40:
                pred_40[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_40[y] = c
            if flag_45:
                pred_45[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_45[y] = c
            if flag_50:
                pred_50[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_50[y] = c
            if flag_55:
                pred_55[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_55[y] = c
            if flag_60:
                pred_60[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_60[y] = c
            if flag_65:
                pred_65[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_65[y] = c
            if flag_70:
                pred_70[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_70[y] = c
            if flag_75:
                pred_75[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_75[y] = c
            if flag_80:
                pred_80[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_80[y] = c
            if flag_85:
                pred_85[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_85[y] = c
            if flag_90:
                pred_90[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_90[y] = c
            if flag_95:
                pred_95[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_95[y] = c
            if flag_99:
                pred_99[y] = predicted_class.numpy()
                if predicted_class == y:
                    e_99[y] = c

        print("!@#")

        print(f"{pred_20},")
        print(f"{pred_25},")
        print(f"{pred_30},")
        print(f"{pred_35},")
        print(f"{pred_40},")
        print(f"{pred_45},")
        print(f"{pred_50},")
        print(f"{pred_55},")
        print(f"{pred_60},")
        print(f"{pred_65},")
        print(f"{pred_70},")
        print(f"{pred_75},")
        print(f"{pred_80},")
        print(f"{pred_85},")
        print(f"{pred_90},")
        print(f"{pred_95},")
        print(f"{pred_99}")

        print("!@#")

        print("@#$")
        
        print(f"{e_20},")
        print(f"{e_25},")
        print(f"{e_30},")
        print(f"{e_35},")
        print(f"{e_40},")
        print(f"{e_45},")
        print(f"{e_50},")
        print(f"{e_55},")
        print(f"{e_60},")
        print(f"{e_65},")
        print(f"{e_70},")
        print(f"{e_75},")
        print(f"{e_80},")
        print(f"{e_85},")
        print(f"{e_90},")
        print(f"{e_95},")
        print(f"{e_99}")
        
        print("@#$")

        return

class EarlyStoppingCallback:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.stopped_early = False
        self.best_weights = None
        
    def should_stop(self, val_loss, model):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            self.best_weights = model.get_weights()
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.stopped_early = True

        return self.stopped_early
    
    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.set_weights(self.best_weights)
        return

def train(model, train_ds, val_ds, x_test, t_test, m_test, y_test, lr, patience, num_classes, filename):

    for inputs, labels in train_ds:
        number = len(inputs.keys())
        break
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    if number == 1:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    else:
        loss_fn = custom_loss

    early_stopping = EarlyStoppingCallback(patience=patience)
    found = Found(x_test, t_test, m_test, y_test, num_classes)

    epochs = 2000
    #history = {'loss': [], 'val_loss': []}

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_accuracy.reset_state()
        val_accuracy.reset_state()

        train_loss = 0
        for step, (batch_train, y_batch_train) in enumerate(train_ds):

            with tf.GradientTape() as tape:
                confs = model(batch_train, training=True)
                if number == 3:
                    x_batch_train = batch_train['flows']
                    t_batch_train = batch_train['timestamps']
                    m_batch_train = batch_train['masks']
                    loss = loss_fn(y_batch_train, confs, m_batch_train)
                elif number == 2:
                    x_batch_train = batch_train['flows']
                    m_batch_train = batch_train['masks']
                    loss = loss_fn(y_batch_train, confs, m_batch_train)
                else:
                    loss = loss_fn(y_batch_train, confs)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_loss += loss.numpy()
            train_accuracy.update_state(y_batch_train, confs)

        train_loss /= len(train_ds)
        #history['loss'].append(train_loss)

        val_loss = 0
        for step, (batch_val, y_batch_val) in enumerate(val_ds):
            confs = model(batch_val, training=False)
            if number == 3:
                x_batch_val = batch_val['flows']
                t_batch_val = batch_val['timestamps']
                m_batch_val = batch_val['masks']
                loss = loss_fn(y_batch_val, confs, m_batch_val)
            elif number == 2:
                x_batch_val = batch_val['flows']
                m_batch_val = batch_val['masks']
                loss = loss_fn(y_batch_val, confs, m_batch_val)
            else:
                loss = loss_fn(y_batch_val, confs)
            val_loss += loss.numpy()
            val_accuracy.update_state(y_batch_val, confs)
        val_loss /= len(val_ds)
        #history['val_loss'].append(val_loss)

        print(f"Train Accuracy: {train_accuracy.result().numpy():.4f}")
        print(f"Train Loss: {train_loss:.12f}")
        print(f"Validation Accuracy: {val_accuracy.result().numpy():.4f}")
        print(f"Validation Loss: {val_loss:.12f}")
        #print(earliness(model, x_test, t_test, m_test, y_test))
        
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            print(f"Elapsed Time: {int(elapsed_time)} seconds")
        elif elapsed_time < 3600:
            minutes = elapsed_time // 60
            seconds = elapsed_time % 60
            print(f"Elapsed Time: {int(minutes)} minutes {int(seconds)} seconds")
        else:
            hours = elapsed_time // 3600
            minutes = (elapsed_time % 3600) // 60
            seconds = elapsed_time % 60
            print(f"Elapsed Time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")

        #found.on_epoch_end(epoch, model)

        if early_stopping and early_stopping.should_stop(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            early_stopping.restore_best_weights(model)
            
            #found.on_training_end(model)
            
            model.save(filename)
            print(f"Model saved with name {filename}")
            
            convert_eval_tflite(model, filename, val_ds, num_classes, x_test, t_test, m_test, y_test)
            break
    
    return

def earliness(model, x_test, t_test, m_test, y_test):

    cur_class = -1
    
    num_samples = x_test.shape[0]
    
    earliness = []
    
    s = ""
    
    for x, t, m, y in zip(x_test, t_test, m_test, y_test):
    
        if y != cur_class:
            s = s + "\n"
            cur_class = y
        
        counter = int(np.sum(m))
        
        augmented_x = [x[:k] for k in range(1, counter+1)]
        augmented_t = [t[:k] for k in range(1, counter+1)]
        augmented_m = [m[:k] for k in range(1, counter+1)]
        
        c = 0
        for x_, t_, m_ in zip(augmented_x, augmented_t, augmented_m):
            
            c = c + 1
            
            x_ = tf.expand_dims(x_, axis=0)
            t_ = tf.expand_dims(t_, axis=0)
            m_ = tf.expand_dims(m_, axis=0)
            
            pred = model.predict((x_, t_, m_), verbose=0)
            pred = tf.squeeze(pred)
            
            top_conf = tf.math.reduce_max(pred)
            predicted_class = tf.argmax(pred)
            
            if predicted_class == y:
                earliness.append(c)
                s = s + f"Class {y} - Packet #{c}: {predicted_class} ({top_conf:.6f})\n"
                break
    
    
    if earliness == []:
        s = s + "Predicted zero samples correctly\n"
    else:
        #s = s + f"Found {len(earliness)} / {num_samples} samples\n"
        s = s + f"\nMean earliness: {np.mean(earliness):.3f}\n"
        s = s + f"Max earliness: {int(np.max(earliness))}\n"
    return s

def print_everything(model, x_test, t_test, m_test, y_test):

    s = ""
    
    for x, t, m, y in zip(x_test, t_test, m_test, y_test):
        
        s = s + f"{y}\n"
        
        counter = int(np.sum(m))
        
        augmented_x = [x[:k] for k in range(1, counter+1)]
        augmented_t = [t[:k] for k in range(1, counter+1)]
        augmented_m = [m[:k] for k in range(1, counter+1)]
        
        c = 0
        for x_, t_, m_ in zip(augmented_x, augmented_t, augmented_m):
            
            c = c + 1
            
            x_ = tf.expand_dims(x_, axis=0)
            t_ = tf.expand_dims(t_, axis=0)
            m_ = tf.expand_dims(m_, axis=0)
            
            pred = model.predict((x_, t_, m_), verbose=0)
            
            pred = tf.squeeze(pred)

            s = s + f"{c}: {tf.argmax(pred)} ({tf.math.reduce_max(pred):.6f})\n"
            
        s = s + "\n"
    
    return s

def threshold_based_eval(model, x_test, t_test, m_test, y_test, threshold, verbose=False):

    accuracy = 0
    earliness = []
    
    cur_class = -1
    
    s = ""
    
    for x, t, m, y in zip(x_test, t_test, m_test, y_test):
    
        if y != cur_class:
            if verbose:
                s = s + "\n"
            cur_class = y
        
        counter = int(np.sum(m))
        
        augmented_x = [x[:k] for k in range(1, counter+1)]
        augmented_t = [t[:k] for k in range(1, counter+1)]
        augmented_m = [m[:k] for k in range(1, counter+1)]
        
        c = 0
        for x_, t_, m_ in zip(augmented_x, augmented_t, augmented_m):
            
            c = c + 1
    
            x_ = tf.expand_dims(x_, axis=0)
            t_ = tf.expand_dims(t_, axis=0)
            m_ = tf.expand_dims(m_, axis=0)
            
            pred = model.predict((x_, t_, m_), verbose=0)
            pred = tf.squeeze(pred)
            
            top_conf = tf.math.reduce_max(pred)
            
            if top_conf >= threshold:
                predicted_class = tf.argmax(pred)
    
                if y == predicted_class:
                    accuracy = accuracy + 1
                    earliness.append(c)
                    
                if verbose:
                    s = s + f"Class {y} - Packet #{c}: {predicted_class} ({top_conf})\n"
                break
                 
    accuracy = 100 * accuracy / x_test.shape[0]

    s = s + f"Threshold: {threshold}\n"
    s = s + f"- Test set accuracy: {accuracy:.2f} %\n"
    if earliness == []:
        s = s + "Empty earliness\n"
    else:
        s = s + f"- Mean earliness: {np.mean(earliness):.3f}\n"
        s = s + f"- Max earliness: {int(np.max(earliness))}\n"
    
    return s

def adjust_timestamps(flows):
    
    adjusted_flows = []
    
    for flow in flows:
        t_0 = float(flow[0].time)
        new_flow = []
        for packet in flow:
            pkt = packet.copy()
            pkt.time = float(packet.time)-t_0
            new_flow.append(pkt)
        adjusted_flows.append(new_flow)

    return adjusted_flows

def create_subflows(flows, k=1):

    subflows = []
    for flow in flows:
        length = len(flow)
        for i in range(1, length+1, k):
            subflows.append(flow[:i])

    return subflows

def preprocess(flows, packet_length, max_num_packets):

    samples_ = []
    times_ = []
    
    for flow in flows:
        
        samples = []
        times = []
        
        for i, packet in enumerate(flow):

            pkt = packet.copy()

            ip_packet = pkt[IP]
            ip_header = ip_packet.copy()
            ip_header.remove_payload()
            
            sample = list(bytes(ip_header))[:10]
            
            tcp_packet = ip_packet.payload
            tcp_header = tcp_packet.copy()
            tcp_header.remove_payload()
    
            s = list(bytes(tcp_header))
    
            sample = sample + s[:16] + s[18:] # remove just the checksum
            
            http_packet = tcp_packet.payload
    
            sample = sample + list(bytes(http_packet))
                
            sample = [x / 255.0 for x in sample]

            if len(sample) < packet_length:
                sample = sample + [0]*(packet_length - len(sample))
            else:
                sample = sample[:packet_length]

            samples.append(sample)
            times.append(float(pkt.time))
            
        samples_.append(samples)
        times_.append(times)

    padded_packets = tf.keras.utils.pad_sequences(
        samples_,
        maxlen=max_num_packets,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )
    
    padded_timestamps = tf.keras.utils.pad_sequences(
        times_,
        maxlen=max_num_packets,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )
    
    attention_masks = tf.cast(padded_packets[..., 0] != 0, dtype=tf.float32)
    
    return padded_packets, padded_timestamps, attention_masks.numpy()

def augment_data(data, label):

    def scale_time(time):
        return base_time + (time - base_time) * scaling_factor

    x = data['flows']
    t = data['timestamps']
    m = data['masks']
    
    max_num_packets = x.shape[0]
    flow_length = tf.reduce_sum(m)
    flow_length = tf.cast(flow_length, tf.int32)

    # t: (30,)
    if flow_length > 1:
        # jitter insertion
        max_fraction = 0.7
        perturbed_times = tf.zeros([max_num_packets], dtype=tf.float32)
        for i in range(1, flow_length):
            
            if i == flow_length-1:
                time_diff = t[i]-t[i-1]
            else:
                t1 = t[i]-t[i-1]
                t2 = t[i+1]-t[i]
                time_diff = tf.minimum(t1, t2)

            max_perturbation = time_diff * max_fraction
            perturbation = tf.random.uniform([], -max_perturbation, max_perturbation)

            perturb_value = t[i] + perturbation
            perturbed_times = tf.tensor_scatter_nd_update(perturbed_times, [[i]], [perturb_value])

        # traffic scaling
        scaling_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
        scaling_factors_tensor = tf.constant(scaling_factors, dtype=tf.float32)
        random_index = tf.random.uniform([], minval=0, maxval=len(scaling_factors), dtype=tf.int32)
        scaling_factor = tf.gather(scaling_factors_tensor, random_index)
        base_time = perturbed_times[0]
        new_timestamps = tf.map_fn(scale_time, perturbed_times, dtype=tf.float32)
    else:
        base_time = tf.constant(0.0, dtype=tf.float32)
        scaling_factor = tf.constant(2.0, dtype=tf.float32)
        new_timestamps = t
        
    # x: (30, 448)
    # drop packets
    flow_length = tf.cast(flow_length, tf.float32)
    max_packets_to_drop = tf.cast(tf.exp(0.07 * flow_length) - 0.5, tf.int32)
    num_packets_to_drop = tf.random.uniform(shape=(), minval=0, maxval=max_packets_to_drop + 1, dtype=tf.int32)
    flow_length = tf.cast(flow_length, tf.int32)
    
    if num_packets_to_drop > 0:

        indices = tf.range(flow_length)
        shuffled_indices = tf.random.shuffle(indices)
        indices_to_keep = shuffled_indices[:flow_length - num_packets_to_drop]
        indices_to_keep = tf.sort(indices_to_keep)
        kept_packets = tf.gather(x, indices_to_keep, axis=0)

        padding = tf.zeros((x.shape[0]-flow_length+num_packets_to_drop, x.shape[1]), dtype=x.dtype)
        padded_flow = tf.concat([kept_packets, padding], axis=0)
        
        kept_timestamps = tf.gather(new_timestamps, indices_to_keep, axis=0)
        padding = tf.zeros((new_timestamps.shape[0]-flow_length+num_packets_to_drop), dtype=new_timestamps.dtype)
        padded_timestamps = tf.concat([kept_timestamps, padding], axis=0)

    else:
        padded_flow = x
        padded_timestamps = new_timestamps

    attention_mask = tf.cast(padded_flow[..., 0] != 0, dtype=tf.float32)
    flow_length = tf.reduce_sum(attention_mask)
    flow_length = tf.cast(flow_length, tf.int32)

    # insert zero packets
    flow_length = tf.cast(flow_length, tf.float32)
    max_packets_to_insert = tf.cast(tf.exp(0.058 * flow_length) - 0.7, tf.int32)
    num_packets_to_insert = tf.random.uniform(shape=(), minval=0, maxval=max_packets_to_insert + 1, dtype=tf.int32)
    flow_length = tf.cast(flow_length, tf.int32)

    if num_packets_to_insert > 0:
        
        indices = tf.range(flow_length)
        shuffled_indices = tf.random.shuffle(indices)
        insert_positions = shuffled_indices[:num_packets_to_insert]
        insert_positions = tf.sort(insert_positions)
        reversed_positions = tf.reverse(insert_positions, axis=[0])

        zero_packet = tf.zeros((1, x.shape[1]), dtype=padded_flow.dtype)  # Shape of one zero packet
        augmented_flow = tf.identity(padded_flow)
        augmented_timestamps = tf.identity(padded_timestamps)
        
        for pos in reversed_positions:

            part1 = augmented_flow[:pos, :]
            part2 = augmented_flow[pos:, :]
            augmented_flow = tf.concat([part1, zero_packet, part2], axis=0)

            part1 = augmented_timestamps[:pos]
            part2 = augmented_timestamps[pos:]
            t_new = tf.convert_to_tensor((augmented_timestamps[pos-1] + augmented_timestamps[pos]) / 2)
            augmented_timestamps = tf.concat([part1, tf.stack([t_new]), part2], axis=0)
        
        final_flow = augmented_flow[:max_num_packets, :]
        final_timestamps = augmented_timestamps[:max_num_packets]

    else:
        final_flow = padded_flow
        final_timestamps = padded_timestamps

    attention_mask = tf.cast(final_flow[..., 0] != 0, dtype=tf.float32)

    # add noise
    noise_stddev = 0.1  # standard deviation for the noise
    num_packets_to_modify = tf.random.uniform(shape=(), minval=0, maxval=10, dtype=tf.int32)
    num_bytes_to_modify = tf.random.uniform(shape=(), minval=0, maxval=5, dtype=tf.int32)

    if num_packets_to_modify > 0 and num_bytes_to_modify > 0:

        noise = tf.random.normal(shape=(num_packets_to_modify, num_bytes_to_modify), mean=0.0, stddev=noise_stddev)
    
        indices_to_modify = tf.random.shuffle(tf.range(tf.shape(final_flow)[0]))[:num_packets_to_modify]
        tiled_tensor1 = tf.tile(tf.reshape(indices_to_modify, (-1, 1)), [1, num_bytes_to_modify])
        byte_indices_to_modify = tf.random.uniform(
            shape=(num_packets_to_modify, num_bytes_to_modify), 
            minval=0, 
            maxval=tf.shape(final_flow)[1], 
            dtype=tf.int32
        )
    
        pairs = tf.stack([tiled_tensor1, byte_indices_to_modify], axis=-1)
        pairs = tf.reshape(pairs, (-1, 2))
    
        final_flow_ = tf.tensor_scatter_nd_update(final_flow, pairs, tf.reshape(noise, [-1]))
    else:
        final_flow_ = final_flow
        
    data['flows'] = final_flow_
    data['timestamps'] = final_timestamps
    data['masks'] = attention_mask
    
    return data, label

def augment_data_xm(data, label):

    def scale_time(time):
        return base_time + (time - base_time) * scaling_factor

    x = data['flows']
    m = data['masks']
    
    max_num_packets = x.shape[0]
    flow_length = tf.reduce_sum(m)
    flow_length = tf.cast(flow_length, tf.int32)

    # x: (30, 448)
    # drop packets
    flow_length = tf.cast(flow_length, tf.float32)
    max_packets_to_drop = tf.cast(tf.exp(0.07 * flow_length) - 0.5, tf.int32)
    num_packets_to_drop = tf.random.uniform(shape=(), minval=0, maxval=max_packets_to_drop + 1, dtype=tf.int32)
    flow_length = tf.cast(flow_length, tf.int32)
    
    if num_packets_to_drop > 0:
        indices = tf.range(flow_length)
        shuffled_indices = tf.random.shuffle(indices)
        indices_to_keep = shuffled_indices[:flow_length - num_packets_to_drop]
        indices_to_keep = tf.sort(indices_to_keep)
        kept_packets = tf.gather(x, indices_to_keep, axis=0)
        padding = tf.zeros((x.shape[0]-flow_length+num_packets_to_drop, x.shape[1]), dtype=x.dtype)
        padded_flow = tf.concat([kept_packets, padding], axis=0)
    else:
        padded_flow = x

    attention_mask = tf.cast(padded_flow[..., 0] != 0, dtype=tf.float32)
    flow_length = tf.reduce_sum(attention_mask)
    flow_length = tf.cast(flow_length, tf.int32)

    # insert zero packets
    flow_length = tf.cast(flow_length, tf.float32)
    max_packets_to_insert = tf.cast(tf.exp(0.058 * flow_length) - 0.7, tf.int32)
    num_packets_to_insert = tf.random.uniform(shape=(), minval=0, maxval=max_packets_to_insert + 1, dtype=tf.int32)
    flow_length = tf.cast(flow_length, tf.int32)

    if num_packets_to_insert > 0:
        indices = tf.range(flow_length)
        shuffled_indices = tf.random.shuffle(indices)
        insert_positions = shuffled_indices[:num_packets_to_insert]
        insert_positions = tf.sort(insert_positions)
        reversed_positions = tf.reverse(insert_positions, axis=[0])

        zero_packet = tf.zeros((1, x.shape[1]), dtype=padded_flow.dtype)  # Shape of one zero packet
        augmented_flow = tf.identity(padded_flow)
        
        for pos in reversed_positions:
            part1 = augmented_flow[:pos, :]
            part2 = augmented_flow[pos:, :]
            augmented_flow = tf.concat([part1, zero_packet, part2], axis=0)
        
        final_flow = augmented_flow[:max_num_packets, :]
    else:
        final_flow = padded_flow

    attention_mask = tf.cast(final_flow[..., 0] != 0, dtype=tf.float32)

    # add noise
    noise_stddev = 0.1  # standard deviation for the noise
    num_packets_to_modify = tf.random.uniform(shape=(), minval=0, maxval=10, dtype=tf.int32)
    num_bytes_to_modify = tf.random.uniform(shape=(), minval=0, maxval=5, dtype=tf.int32)

    if num_packets_to_modify > 0 and num_bytes_to_modify > 0:

        noise = tf.random.normal(shape=(num_packets_to_modify, num_bytes_to_modify), mean=0.0, stddev=noise_stddev)
    
        indices_to_modify = tf.random.shuffle(tf.range(tf.shape(final_flow)[0]))[:num_packets_to_modify]
        tiled_tensor1 = tf.tile(tf.reshape(indices_to_modify, (-1, 1)), [1, num_bytes_to_modify])
        byte_indices_to_modify = tf.random.uniform(
            shape=(num_packets_to_modify, num_bytes_to_modify), 
            minval=0, 
            maxval=tf.shape(final_flow)[1], 
            dtype=tf.int32
        )
    
        pairs = tf.stack([tiled_tensor1, byte_indices_to_modify], axis=-1)
        pairs = tf.reshape(pairs, (-1, 2))
    
        final_flow_ = tf.tensor_scatter_nd_update(final_flow, pairs, tf.reshape(noise, [-1]))
    else:
        final_flow_ = final_flow
        
    data['flows'] = final_flow_
    data['masks'] = attention_mask
    
    return data, label

def augment_data_x(data, label):

    def scale_time(time):
        return base_time + (time - base_time) * scaling_factor
        
    @tf.function
    def find_flow_length(x, max_num_packets):
        x = tf.reduce_sum(x, axis=1)
        zero_indices = tf.where(x == 0)
        if tf.size(zero_indices) > 0:
            return tf.cast(zero_indices[0][0], tf.int32)
        else:
            return max_num_packets
    
    x = data['flows']
    
    max_num_packets = x.shape[0]
    flow_length = find_flow_length(x, max_num_packets)
    
    # x: (30, 448)
    # drop packets
    flow_length = tf.cast(flow_length, tf.float32)
    max_packets_to_drop = tf.cast(tf.exp(0.07 * flow_length) - 0.5, tf.int32)
    num_packets_to_drop = tf.random.uniform(shape=(), minval=0, maxval=max_packets_to_drop + 1, dtype=tf.int32)
    flow_length = tf.cast(flow_length, tf.int32)
    
    if num_packets_to_drop > 0:
        indices = tf.range(flow_length)
        shuffled_indices = tf.random.shuffle(indices)
        indices_to_keep = shuffled_indices[:flow_length - num_packets_to_drop]
        indices_to_keep = tf.sort(indices_to_keep)
        kept_packets = tf.gather(x, indices_to_keep, axis=0)
        padding = tf.zeros((x.shape[0]-flow_length+num_packets_to_drop, x.shape[1]), dtype=x.dtype)
        padded_flow = tf.concat([kept_packets, padding], axis=0)
    else:
        padded_flow = x

    flow_length = find_flow_length(padded_flow, max_num_packets)

    # insert zero packets
    flow_length = tf.cast(flow_length, tf.float32)
    max_packets_to_insert = tf.cast(tf.exp(0.058 * flow_length) - 0.7, tf.int32)
    num_packets_to_insert = tf.random.uniform(shape=(), minval=0, maxval=max_packets_to_insert + 1, dtype=tf.int32)
    flow_length = tf.cast(flow_length, tf.int32)

    if num_packets_to_insert > 0:
        indices = tf.range(flow_length)
        shuffled_indices = tf.random.shuffle(indices)
        insert_positions = shuffled_indices[:num_packets_to_insert]
        insert_positions = tf.sort(insert_positions)
        reversed_positions = tf.reverse(insert_positions, axis=[0])

        zero_packet = tf.zeros((1, x.shape[1]), dtype=padded_flow.dtype)  # Shape of one zero packet
        augmented_flow = tf.identity(padded_flow)
        
        for pos in reversed_positions:
            part1 = augmented_flow[:pos, :]
            part2 = augmented_flow[pos:, :]
            augmented_flow = tf.concat([part1, zero_packet, part2], axis=0)
        
        final_flow = augmented_flow[:max_num_packets, :]
    else:
        final_flow = padded_flow
    
    # add noise
    noise_stddev = 0.1  # standard deviation for the noise
    num_packets_to_modify = tf.random.uniform(shape=(), minval=0, maxval=10, dtype=tf.int32)
    num_bytes_to_modify = tf.random.uniform(shape=(), minval=0, maxval=5, dtype=tf.int32)

    if num_packets_to_modify > 0 and num_bytes_to_modify > 0:

        noise = tf.random.normal(shape=(num_packets_to_modify, num_bytes_to_modify), mean=0.0, stddev=noise_stddev)
    
        indices_to_modify = tf.random.shuffle(tf.range(tf.shape(final_flow)[0]))[:num_packets_to_modify]
        tiled_tensor1 = tf.tile(tf.reshape(indices_to_modify, (-1, 1)), [1, num_bytes_to_modify])
        byte_indices_to_modify = tf.random.uniform(
            shape=(num_packets_to_modify, num_bytes_to_modify), 
            minval=0, 
            maxval=tf.shape(final_flow)[1], 
            dtype=tf.int32
        )
    
        pairs = tf.stack([tiled_tensor1, byte_indices_to_modify], axis=-1)
        pairs = tf.reshape(pairs, (-1, 2))
    
        final_flow_ = tf.tensor_scatter_nd_update(final_flow, pairs, tf.reshape(noise, [-1]))
    else:
        final_flow_ = final_flow
    
    data['flows'] = final_flow_
    
    return data, label

def to_ternary(n, num_classes):

    ternary = []
    while n > 0:
        ternary.append(n % 3)
        n //= 3

    while len(ternary) < num_classes:
        ternary.append(0)
    
    return ternary[::-1]

def get_dataset(root_dir, packet_length, max_num_packets, batch_size, num_classes, split, expand_factor):

    flows_sql, flows_command, flows_backdoor, flows_uploading, flows_xss, flows_high, flows_benign = read_files_v2(root_dir, max_num_packets)

    flows_sql_ = [values[:max_num_packets] for values in flows_sql.values()]
    flows_command_ = [values[:max_num_packets] for values in flows_command.values()]
    flows_backdoor_ = [values[:max_num_packets] for values in flows_backdoor.values()]
    flows_uploading_ = [values[:max_num_packets] for values in flows_uploading.values()]
    flows_xss_ = [values[:max_num_packets] for values in flows_xss.values()]
    if num_classes == 7:
        flows_high_ = [values[:max_num_packets] for values in flows_high.values()]
    flows_benign_ = [values[:max_num_packets] for values in flows_benign.values()]

    flows_sql_ = adjust_timestamps(flows_sql_)
    flows_command_ = adjust_timestamps(flows_command_)
    flows_backdoor_ = adjust_timestamps(flows_backdoor_)
    flows_uploading_ = adjust_timestamps(flows_uploading_)
    flows_xss_ = adjust_timestamps(flows_xss_)
    if num_classes == 7:
        flows_high_ = adjust_timestamps(flows_high_)
    flows_benign_ = adjust_timestamps(flows_benign_)
    
    splits = to_ternary(split, num_classes)
    #print(splits)
    
    if splits[0] == 0:
        flows_sql_train = flows_sql_[:2]
        flows_sql_test = flows_sql_[2:]
    elif splits[0] == 1:
        flows_sql_train = flows_sql_[1:]
        flows_sql_test = flows_sql_[0:1]
    else: # 2
        flows_sql_train = flows_sql_[::2]
        flows_sql_test = flows_sql_[1:2]
    if splits[1] == 0:
        flows_command_train = flows_command_[:2]
        flows_command_test = flows_command_[2:]
    elif splits[1] == 1:
        flows_command_train = flows_command_[1:]
        flows_command_test = flows_command_[0:1]
    else: # 2
        flows_command_train = flows_command_[::2]
        flows_command_test = flows_command_[1:2]
    if splits[2] == 0:
        flows_backdoor_train = flows_backdoor_[:2]
        flows_backdoor_test = flows_backdoor_[2:]
    elif splits[2] == 1:
        flows_backdoor_train = flows_backdoor_[1:]
        flows_backdoor_test = flows_backdoor_[0:1]
    else: # 2
        flows_backdoor_train = flows_backdoor_[::2]
        flows_backdoor_test = flows_backdoor_[1:2]
    if splits[3] == 0:
        flows_uploading_train = flows_uploading_[:2]
        flows_uploading_test = flows_uploading_[2:]
    elif splits[3] == 1:
        flows_uploading_train = flows_uploading_[1:]
        flows_uploading_test = flows_uploading_[0:1]
    else: # 2
        flows_uploading_train = flows_uploading_[::2]
        flows_uploading_test = flows_uploading_[1:2]
    if splits[4] == 0:
        flows_xss_train = flows_xss_[:2]
        flows_xss_test = flows_xss_[2:]
    elif splits[4] == 1:
        flows_xss_train = flows_xss_[1:]
        flows_xss_test = flows_xss_[0:1]
    else: # 2
        flows_xss_train = flows_xss_[::2]
        flows_xss_test = flows_xss_[1:2]
    if num_classes == 7:
        if splits[5] == 0:
            flows_high_train = flows_high_[:2]
            flows_high_test = flows_high_[2:]
        elif splits[5] == 1:
            flows_high_train = flows_high_[1:]
            flows_high_test = flows_high_[0:1]
        else: # 2
            flows_high_train = flows_high_[::2]
            flows_high_test = flows_high_[1:2]
        if splits[6] == 0:
            flows_benign_train = flows_benign_[:2]
            flows_benign_test = flows_benign_[2:]
        elif splits[6] == 1:
            flows_benign_train = flows_benign_[1:]
            flows_benign_test = flows_benign_[0:1]
        else: # 2
            flows_benign_train = flows_benign_[::2]
            flows_benign_test = flows_benign_[1:2]
    else:
        if splits[5] == 0:
            flows_benign_train = flows_benign_[:2]
            flows_benign_test = flows_benign_[2:]
        elif splits[5] == 1:
            flows_benign_train = flows_benign_[1:]
            flows_benign_test = flows_benign_[0:1]
        else: # 2
            flows_benign_train = flows_benign_[::2]
            flows_benign_test = flows_benign_[1:2]
    
    # if split == "A":
    #     flows_sql_train = flows_sql_[:2]
    #     flows_command_train = flows_command_[:2]
    #     flows_backdoor_train = flows_backdoor_[:2]
    #     flows_uploading_train = flows_uploading_[:2]
    #     flows_xss_train = flows_xss_[:2]
    #     if num_classes == 7:
    #         flows_high_train = flows_high_[:2]
    #     flows_benign_train = flows_benign_[:2]
    # elif split == "B":
    #     flows_sql_train = flows_sql_[1:]
    #     flows_command_train = flows_command_[1:]
    #     flows_backdoor_train = flows_backdoor_[1:]
    #     flows_uploading_train = flows_uploading_[1:]
    #     flows_xss_train = flows_xss_[1:]
    #     if num_classes == 7:
    #         flows_high_train = flows_high_[1:]
    #     flows_benign_train = flows_benign_[1:]
    # elif split == "C":
    #     flows_sql_train = flows_sql_[::2]
    #     flows_command_train= flows_command_[::2]
    #     flows_backdoor_train = flows_backdoor_[::2]
    #     flows_uploading_train = flows_uploading_[::2]
    #     flows_xss_train = flows_xss_[::2]
    #     if num_classes == 7:
    #         flows_high_train = flows_high_[::2]
    #     flows_benign_train = flows_benign_[::2]
    # 
    # if split == "A":
    #     flows_sql_test = flows_sql_[2:]
    #     flows_command_test = flows_command_[2:]
    #     flows_backdoor_test = flows_backdoor_[2:]
    #     flows_uploading_test = flows_uploading_[2:]
    #     flows_xss_test = flows_xss_[2:]
    #     if num_classes == 7:
    #         flows_high_test = flows_high_[2:]
    #     flows_benign_test = flows_benign_[2:]
    # elif split == "B":
    #     flows_sql_test = flows_sql_[0:1]
    #     flows_command_test = flows_command_[0:1]
    #     flows_backdoor_test = flows_backdoor_[0:1]
    #     flows_uploading_test = flows_uploading_[0:1]
    #     flows_xss_test = flows_xss_[0:1]
    #     if num_classes == 7:
    #         flows_high_test = flows_high_[0:1]
    #     flows_benign_test = flows_benign_[0:1]
    # elif split == "C":
    #     flows_sql_test = flows_sql_[1:2]
    #     flows_command_test = flows_command_[1:2]
    #     flows_backdoor_test = flows_backdoor_[1:2]
    #     flows_uploading_test = flows_uploading_[1:2]
    #     flows_xss_test = flows_xss_[1:2]
    #     if num_classes == 7:
    #         flows_high_test = flows_high_[1:2]
    #     flows_benign_test = flows_benign_[1:2]

    flows_sql_sub = create_subflows(flows_sql_train)
    flows_command_sub = create_subflows(flows_command_train)
    flows_backdoor_sub = create_subflows(flows_backdoor_train)
    flows_uploading_sub = create_subflows(flows_uploading_train)
    flows_xss_sub = create_subflows(flows_xss_train)
    if num_classes == 7:
        flows_high_sub = create_subflows(flows_high_train)
    flows_benign_sub = create_subflows(flows_benign_train)

    d_sql, t_sql, m_sql = preprocess(flows_sql_sub, packet_length, max_num_packets)
    d_command, t_command, m_command = preprocess(flows_command_sub, packet_length, max_num_packets)
    d_backdoor, t_backdoor, m_backdoor = preprocess(flows_backdoor_sub, packet_length, max_num_packets)
    d_uploading, t_uploading, m_uploading = preprocess(flows_uploading_sub, packet_length, max_num_packets)
    d_xss, t_xss, m_xss = preprocess(flows_xss_sub, packet_length, max_num_packets)
    if num_classes == 7:
        d_high, t_high, m_high = preprocess(flows_high_sub, packet_length, max_num_packets)
    d_benign, t_benign, m_benign = preprocess(flows_benign_sub, packet_length, max_num_packets)

    if num_classes == 7:
        min_number = np.min([d_sql.shape[0], d_command.shape[0], d_backdoor.shape[0], d_uploading.shape[0], d_xss.shape[0], d_high.shape[0], d_benign.shape[0]])
    else:
        min_number = np.min([d_sql.shape[0], d_command.shape[0], d_backdoor.shape[0], d_uploading.shape[0], d_xss.shape[0], d_benign.shape[0]])
    #print(min_number)

    d_sql_aug, t_sql_aug, m_sql_aug = randomly_keep_elements(d_sql, t_sql, m_sql, min_number)
    d_command_aug, t_command_aug, m_command_aug = randomly_keep_elements(d_command, t_command, m_command, min_number)
    d_backdoor_aug, t_backdoor_aug, m_backdoor_aug = randomly_keep_elements(d_backdoor, t_backdoor, m_backdoor, min_number)
    d_uploading_aug, t_uploading_aug, m_uploading_aug = randomly_keep_elements(d_uploading, t_uploading, m_uploading, min_number)
    d_xss_aug, t_xss_aug, m_xss_aug = randomly_keep_elements(d_xss, t_xss, m_xss, min_number)
    if num_classes == 7:
        d_high_aug, t_high_aug, m_high_aug = randomly_keep_elements(d_high, t_high, m_high, min_number)
    d_benign_aug, t_benign_aug, m_benign_aug = randomly_keep_elements(d_benign, t_benign, m_benign, min_number)

    d_sql_aug_ = np.tile(d_sql_aug, (expand_factor, 1, 1))
    d_command_aug_ = np.tile(d_command_aug, (expand_factor, 1, 1))
    d_backdoor_aug_ = np.tile(d_backdoor_aug, (expand_factor, 1, 1))
    d_uploading_aug_ = np.tile(d_uploading_aug, (expand_factor, 1, 1))
    d_xss_aug_ = np.tile(d_xss_aug, (expand_factor, 1, 1))
    if num_classes == 7:
        d_high_aug_ = np.tile(d_high_aug, (expand_factor, 1, 1))
    d_benign_aug_ = np.tile(d_benign_aug, (expand_factor, 1, 1))

    t_sql_aug_ = np.tile(t_sql_aug, (expand_factor, 1))
    t_command_aug_ = np.tile(t_command_aug, (expand_factor, 1))
    t_backdoor_aug_ = np.tile(t_backdoor_aug, (expand_factor, 1))
    t_uploading_aug_ = np.tile(t_uploading_aug, (expand_factor, 1))
    t_xss_aug_ = np.tile(t_xss_aug, (expand_factor, 1))
    if num_classes == 7:
        t_high_aug_ = np.tile(t_high_aug, (expand_factor, 1))
    t_benign_aug_ = np.tile(t_benign_aug, (expand_factor, 1))

    m_sql_aug_ = np.tile(m_sql_aug, (expand_factor, 1))
    m_command_aug_ = np.tile(m_command_aug, (expand_factor, 1))
    m_backdoor_aug_ = np.tile(m_backdoor_aug, (expand_factor, 1))
    m_uploading_aug_ = np.tile(m_uploading_aug, (expand_factor, 1))
    m_xss_aug_ = np.tile(m_xss_aug, (expand_factor, 1))
    if num_classes == 7:
        m_high_aug_ = np.tile(m_high_aug, (expand_factor, 1))
    m_benign_aug_ = np.tile(m_benign_aug, (expand_factor, 1))

    if num_classes == 7:
        x_train = np.concatenate((d_sql_aug_, d_command_aug_, d_backdoor_aug_, d_uploading_aug_, d_xss_aug_, d_high_aug_, d_benign_aug_), axis=0)
        t_train = np.concatenate((t_sql_aug_, t_command_aug_, t_backdoor_aug_, t_uploading_aug_, t_xss_aug_, t_high_aug_, t_benign_aug_), axis=0)
        m_train = np.concatenate((m_sql_aug_, m_command_aug_, m_backdoor_aug_, m_uploading_aug_, m_xss_aug_, m_high_aug_, m_benign_aug_), axis=0)
        y = d_sql_aug_.shape[0]*[0] + d_command_aug_.shape[0]*[1] + d_backdoor_aug_.shape[0]*[2] + d_uploading_aug_.shape[0]*[3] + d_xss_aug_.shape[0]*[4] + d_high_aug_.shape[0]*[5] + d_benign_aug_.shape[0]*[6]
    else:
        x_train = np.concatenate((d_sql_aug_, d_command_aug_, d_backdoor_aug_, d_uploading_aug_, d_xss_aug_, d_benign_aug_), axis=0)
        t_train = np.concatenate((t_sql_aug_, t_command_aug_, t_backdoor_aug_, t_uploading_aug_, t_xss_aug_, t_benign_aug_), axis=0)
        m_train = np.concatenate((m_sql_aug_, m_command_aug_, m_backdoor_aug_, m_uploading_aug_, m_xss_aug_, m_benign_aug_), axis=0)
        y = d_sql_aug_.shape[0]*[0] + d_command_aug_.shape[0]*[1] + d_backdoor_aug_.shape[0]*[2] + d_uploading_aug_.shape[0]*[3] + d_xss_aug_.shape[0]*[4] + d_benign_aug_.shape[0]*[5]
    y_train = np.array(y)

    x_train, x_val, t_train, t_val, m_train, m_val, y_train, y_val = train_test_split(x_train, t_train, m_train, y_train, test_size=0.05, random_state=42)

    d_sql_test, t_sql_test, m_sql_test = preprocess(flows_sql_test, packet_length, max_num_packets)
    d_command_test, t_command_test, m_command_test = preprocess(flows_command_test, packet_length, max_num_packets)
    d_backdoor_test, t_backdoor_test, m_backdoor_test = preprocess(flows_backdoor_test, packet_length, max_num_packets)
    d_uploading_test, t_uploading_test, m_uploading_test = preprocess(flows_uploading_test, packet_length, max_num_packets)
    d_xss_test, t_xss_test, m_xss_test = preprocess(flows_xss_test, packet_length, max_num_packets)
    if num_classes == 7:
        d_high_test, t_high_test, m_high_test = preprocess(flows_high_test, packet_length, max_num_packets)
    d_benign_test, t_benign_test, m_benign_test = preprocess(flows_benign_test, packet_length, max_num_packets)

    if num_classes == 7:
        x_test = np.concatenate((d_sql_test, d_command_test, d_backdoor_test, d_uploading_test, d_xss_test, d_high_test, d_benign_test), axis=0)
        t_test = np.concatenate((t_sql_test, t_command_test, t_backdoor_test, t_uploading_test, t_xss_test, t_high_test, t_benign_test), axis=0)
        m_test = np.concatenate((m_sql_test, m_command_test, m_backdoor_test, m_uploading_test, m_xss_test, m_high_test, m_benign_test), axis=0)
        y = d_sql_test.shape[0]*[0] + d_command_test.shape[0]*[1] + d_backdoor_test.shape[0]*[2] + d_uploading_test.shape[0]*[3] + d_xss_test.shape[0]*[4] + d_high_test.shape[0]*[5] + d_benign_test.shape[0]*[6]
    else:
        x_test = np.concatenate((d_sql_test, d_command_test, d_backdoor_test, d_uploading_test, d_xss_test, d_benign_test), axis=0)
        t_test = np.concatenate((t_sql_test, t_command_test, t_backdoor_test, t_uploading_test, t_xss_test, t_benign_test), axis=0)
        m_test = np.concatenate((m_sql_test, m_command_test, m_backdoor_test, m_uploading_test, m_xss_test, m_benign_test), axis=0)
        y = d_sql_test.shape[0]*[0] + d_command_test.shape[0]*[1] + d_backdoor_test.shape[0]*[2] + d_uploading_test.shape[0]*[3] + d_xss_test.shape[0]*[4] + d_benign_test.shape[0]*[5]
    y_test = np.array(y)

    print(f"- Split {split}")
    print(f"- Training samples: {x_train.shape[0]}")
    print(f"- Validation samples: {x_val.shape[0]}")
    print(f"- Testing samples: {x_test.shape[0]}")

    # TF dataset
    BUFFER_SIZE = batch_size * 2
    AUTO = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices(({'flows': x_train, 'timestamps': t_train, 'masks': m_train}, y_train))
    train_ds_xm = train_ds.map(lambda inputs, labels: ({'flows': inputs['flows'], 'masks': inputs['masks']}, labels))
    train_ds_x = train_ds.map(lambda inputs, labels: ({'flows': inputs['flows']}, labels))
    #train_ds = tf.data.Dataset.from_tensor_slices(((x_train, t_train, m_train), y_train))
    train_ds = train_ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE).shuffle(BUFFER_SIZE).batch(batch_size).prefetch(AUTO)
    train_ds_xm = train_ds_xm.map(augment_data_xm, num_parallel_calls=tf.data.AUTOTUNE).shuffle(BUFFER_SIZE).batch(batch_size).prefetch(AUTO)
    train_ds_x = train_ds_x.map(augment_data_x, num_parallel_calls=tf.data.AUTOTUNE).shuffle(BUFFER_SIZE).batch(batch_size).prefetch(AUTO)

    val_ds = tf.data.Dataset.from_tensor_slices(({'flows': x_val, 'timestamps': t_val, 'masks': m_val}, y_val))
    val_ds_xm = val_ds.map(lambda inputs, labels: ({'flows': inputs['flows'], 'masks': inputs['masks']}, labels))
    val_ds_x = val_ds.map(lambda inputs, labels: ({'flows': inputs['flows']}, labels))
    #val_ds = tf.data.Dataset.from_tensor_slices(((x_val, t_val, m_val), y_val))
    val_ds = val_ds.batch(batch_size).prefetch(AUTO)
    val_ds_xm = val_ds_xm.batch(batch_size).prefetch(AUTO)
    val_ds_x = val_ds_x.batch(batch_size).prefetch(AUTO)

    return train_ds, train_ds_xm, train_ds_x, val_ds, val_ds_xm, val_ds_x, (x_test, t_test, m_test, y_test)

def convert_eval_tflite(model, filename, val_ds, num_classes, x_test, t_test, m_test, y_test):
    
    for inputs, labels in val_ds:
        number = len(inputs.keys())
        break
    
    filename_fp32 = filename.split(".")[0] + ".tflite"
    filename_int8 = filename.split(".")[0] + "_int.tflite"
    
    # fp32
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if "rnn" in filename:
        converter.experimental_enable_resource_variables = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
    tflite_model = converter.convert()
    with open(filename_fp32, "wb") as f:
        f.write(tflite_model)
    
    # int8
    def representative_data_gen():
        if number == 3:
            for batch, labels in val_ds:
                x = batch['flows']
                t = batch['timestamps']
                m = batch['masks']
                for x_, t_, m_ in zip(x, t, m):
                    x_ = tf.expand_dims(x_, axis=0)
                    t_ = tf.expand_dims(t_, axis=0)
                    m_ = tf.expand_dims(m_, axis=0)
                    yield [x_, t_, m_]
        elif number == 2:
            for batch, label in val_ds:
                x = batch['flows']
                m = batch['masks']
                for x_, m_ in zip(x, m):
                    x_ = tf.expand_dims(x_, axis=0)
                    m_ = tf.expand_dims(m_, axis=0)
                    yield [x_, m_]
        else:
            for batch, label in val_ds:
                x = batch['flows']
                for x_ in x:
                    x_ = tf.expand_dims(x_, axis=0)
                    yield [x_]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    if "rnn" in filename:
        converter.experimental_enable_resource_variables = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
    else:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_model = converter.convert()
    with open(filename_int8, "wb") as f:
        f.write(tflite_model)

    # evaluate
    eval_tflite(filename_fp32, num_classes, x_test, t_test, m_test, y_test)
    eval_tflite(filename_int8, num_classes, x_test, t_test, m_test, y_test)
    
    return
    
def eval_tflite(model_path, num_classes, x_test, t_test, m_test, y_test):

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    number = len(input_details)

    num_samples = x_test.shape[0]
    
    quant = False
    if input_details[0]['dtype'] == np.int8:
        quant = True
        x_scale, x_zero_point = input_details[0]['quantization']
        if number == 3:
            t_scale, t_zero_point = input_details[1]['quantization']
            m_scale, m_zero_point = input_details[2]['quantization']
        if number == 2:
            m_scale, m_zero_point = input_details[1]['quantization']
        output_scale, output_zero_point = output_details[0]['quantization']

    special_p = "@tflite_p_int" if quant else "@tflite_p"
    special_e = "@tflite_e_int" if quant else "@tflite_e"

    pred_20 = [-1]*num_samples
    pred_25 = [-1]*num_samples
    pred_30 = [-1]*num_samples
    pred_35 = [-1]*num_samples
    pred_40 = [-1]*num_samples
    pred_45 = [-1]*num_samples
    pred_50 = [-1]*num_samples
    pred_55 = [-1]*num_samples
    pred_60 = [-1]*num_samples
    pred_65 = [-1]*num_samples
    pred_70 = [-1]*num_samples
    pred_75 = [-1]*num_samples
    pred_80 = [-1]*num_samples
    pred_85 = [-1]*num_samples
    pred_90 = [-1]*num_samples
    pred_95 = [-1]*num_samples
    pred_99 = [-1]*num_samples

    e_20 = [-1]*num_samples
    e_25 = [-1]*num_samples
    e_30 = [-1]*num_samples
    e_35 = [-1]*num_samples
    e_40 = [-1]*num_samples
    e_45 = [-1]*num_samples
    e_50 = [-1]*num_samples
    e_55 = [-1]*num_samples
    e_60 = [-1]*num_samples
    e_65 = [-1]*num_samples
    e_70 = [-1]*num_samples
    e_75 = [-1]*num_samples
    e_80 = [-1]*num_samples
    e_85 = [-1]*num_samples
    e_90 = [-1]*num_samples
    e_95 = [-1]*num_samples
    e_99 = [-1]*num_samples
    print_flag = [True]*num_samples
    
    #cur_class = -1
    
    for i, (x, t, m, y) in enumerate(zip(x_test, t_test, m_test, y_test)):
        
        flag_20 = flag_25 = flag_30 = flag_35 = flag_40 = flag_45 = flag_50 = flag_55 = True
        flag_60 = flag_65 = flag_70 = flag_75 = flag_80 = flag_85 = flag_90 = flag_95 = flag_99 = True
        
        # if y != cur_class:
        #     cur_class = y
        
        counter = int(np.sum(m))
        
        augmented_x = [x[:k] for k in range(1, counter+1)]
        augmented_t = [t[:k] for k in range(1, counter+1)]
        augmented_m = [m[:k] for k in range(1, counter+1)]
        
        c = 0
        for x_, t_, m_ in zip(augmented_x, augmented_t, augmented_m):
            
            c = c + 1

            # quantize
            if quant:
                x_ = x_ / x_scale + x_zero_point
                x_ = np.clip(x_, -128, 127).astype(np.int8)
                if number == 3:
                    t_ = t_ / t_scale + t_zero_point
                    t_ = np.clip(t_, -128, 127).astype(np.int8)
                    m_ = m_ / m_scale + m_zero_point
                    m_ = np.clip(m_, -128, 127).astype(np.int8)
                if number == 2:
                    m_ = m_ / m_scale + m_zero_point
                    m_ = np.clip(m_, -128, 127).astype(np.int8)
            
            x_ = tf.expand_dims(x_, axis=0)
            if number == 3:
                t_ = tf.expand_dims(t_, axis=0)
                m_ = tf.expand_dims(m_, axis=0)
            if number == 2:
                m_ = tf.expand_dims(m_, axis=0)
            
            n = x_.shape[1]
            d = x_.shape[2]

            interpreter.resize_tensor_input(input_details[0]['index'], [1, n, d])
            if number == 3:
                interpreter.resize_tensor_input(input_details[1]['index'], [1, n])
                interpreter.resize_tensor_input(input_details[2]['index'], [1, n])
            if number == 2:
                interpreter.resize_tensor_input(input_details[1]['index'], [1, n])
            interpreter.allocate_tensors()
            
            #pred = model.predict((x_, t_, m_), verbose=0)

            interpreter.set_tensor(input_details[0]['index'], x_)
            if number == 3:
                interpreter.set_tensor(input_details[1]['index'], t_)
                interpreter.set_tensor(input_details[2]['index'], m_)
            if number == 2:
                interpreter.set_tensor(input_details[1]['index'], m_)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # dequantize
            if quant:
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
            
            #print(output_data)
            
            pred = tf.squeeze(output_data)
            #pred = tf.squeeze(pred)
            
            top_conf = tf.math.reduce_max(pred)
            predicted_class = tf.argmax(pred)

            if flag_20 and top_conf >= 0.2:
                pred_20[i] = predicted_class.numpy()
                flag_20 = False
                if predicted_class == y:
                    e_20[i] = c
            
            if flag_25 and top_conf >= 0.25:
                pred_25[i] = predicted_class.numpy()
                flag_25 = False
                if predicted_class == y:
                    e_25[i] = c
            
            if flag_30 and top_conf >= 0.3:
                pred_30[i] = predicted_class.numpy()
                flag_30 = False
                if predicted_class == y:
                    e_30[i] = c
            
            if flag_35 and top_conf >= 0.35:
                pred_35[i] = predicted_class.numpy()
                flag_35 = False
                if predicted_class == y:
                    e_35[i] = c
            
            if flag_40 and top_conf >= 0.4:
                pred_40[i] = predicted_class.numpy()
                flag_40 = False
                if predicted_class == y:
                    e_40[i] = c

            if flag_45 and top_conf >= 0.45:
                pred_45[i] = predicted_class.numpy()
                flag_45 = False
                if predicted_class == y:
                    e_45[i] = c
            
            if flag_50 and top_conf >= 0.5:
                pred_50[i] = predicted_class.numpy()
                flag_50 = False
                if predicted_class == y:
                    e_50[i] = c

            if flag_55 and top_conf >= 0.55:
                pred_55[i] = predicted_class.numpy()
                flag_55 = False
                if predicted_class == y:
                    e_55[i] = c
            
            if flag_60 and top_conf >= 0.6:
                pred_60[i] = predicted_class.numpy()
                flag_60 = False
                if predicted_class == y:
                    e_60[i] = c

            if flag_65 and top_conf >= 0.65:
                pred_65[i] = predicted_class.numpy()
                flag_65 = False
                if predicted_class == y:
                    e_65[i] = c
            
            if flag_70 and top_conf >= 0.7:
                pred_70[i] = predicted_class.numpy()
                flag_70 = False
                if predicted_class == y:
                    e_70[i] = c
            
            if flag_75 and top_conf >= 0.75:
                pred_75[i] = predicted_class.numpy()
                flag_75 = False
                if predicted_class == y:
                    e_75[i] = c

            if flag_80 and top_conf >= 0.8:
                pred_80[i] = predicted_class.numpy()
                flag_80 = False
                if predicted_class == y:
                    e_80[i] = c

            if flag_85 and top_conf >= 0.85:
                pred_85[i] = predicted_class.numpy()
                flag_85 = False
                if predicted_class == y:
                    e_85[i] = c
            
            if flag_90 and top_conf >= 0.9:
                pred_90[i] = predicted_class.numpy()
                flag_90 = False
                if predicted_class == y:
                    e_90[i] = c
            
            if flag_95 and top_conf >= 0.95:
                pred_95[i] = predicted_class.numpy()
                flag_95 = False
                if predicted_class == y:
                    e_95[i] = c
                        
            if flag_99 and top_conf >= 0.99:
                pred_99[i] = predicted_class.numpy()
                flag_99 = False
                if predicted_class == y:
                    e_99[i] = c
            
        if flag_20:
            pred_20[i] = predicted_class.numpy()
            if predicted_class == y:
                e_20[i] = c
        if flag_25:
            pred_25[i] = predicted_class.numpy()
            if predicted_class == y:
                e_25[i] = c
        if flag_30:
            pred_30[i] = predicted_class.numpy()
            if predicted_class == y:
                e_30[i] = c
        if flag_35:
            pred_35[i] = predicted_class.numpy()
            if predicted_class == y:
                e_35[i] = c
        if flag_40:
            pred_40[i] = predicted_class.numpy()
            if predicted_class == y:
                e_40[i] = c
        if flag_45:
            pred_45[i] = predicted_class.numpy()
            if predicted_class == y:
                e_45[i] = c
        if flag_50:
            pred_50[i] = predicted_class.numpy()
            if predicted_class == y:
                e_50[i] = c
        if flag_55:
            pred_55[i] = predicted_class.numpy()
            if predicted_class == y:
                e_55[i] = c
        if flag_60:
            pred_60[i] = predicted_class.numpy()
            if predicted_class == y:
                e_60[i] = c
        if flag_65:
            pred_65[i] = predicted_class.numpy()
            if predicted_class == y:
                e_65[i] = c
        if flag_70:
            pred_70[i] = predicted_class.numpy()
            if predicted_class == y:
                e_70[i] = c
        if flag_75:
            pred_75[i] = predicted_class.numpy()
            if predicted_class == y:
                e_75[i] = c
        if flag_80:
            pred_80[i] = predicted_class.numpy()
            if predicted_class == y:
                e_80[i] = c
        if flag_85:
            pred_85[i] = predicted_class.numpy()
            if predicted_class == y:
                e_85[i] = c
        if flag_90:
            pred_90[i] = predicted_class.numpy()
            if predicted_class == y:
                e_90[i] = c
        if flag_95:
            pred_95[i] = predicted_class.numpy()
            if predicted_class == y:
                e_95[i] = c
        if flag_99:
            pred_99[i] = predicted_class.numpy()
            if predicted_class == y:
                e_99[i] = c

    print(special_p)

    print(f"{pred_20},")
    print(f"{pred_25},")
    print(f"{pred_30},")
    print(f"{pred_35},")
    print(f"{pred_40},")
    print(f"{pred_45},")
    print(f"{pred_50},")
    print(f"{pred_55},")
    print(f"{pred_60},")
    print(f"{pred_65},")
    print(f"{pred_70},")
    print(f"{pred_75},")
    print(f"{pred_80},")
    print(f"{pred_85},")
    print(f"{pred_90},")
    print(f"{pred_95},")
    print(f"{pred_99}")

    print(special_p)

    print(special_e)

    print(f"{e_20},")
    print(f"{e_25},")
    print(f"{e_30},")
    print(f"{e_35},")
    print(f"{e_40},")
    print(f"{e_45},")
    print(f"{e_50},")
    print(f"{e_55},")
    print(f"{e_60},")
    print(f"{e_65},")
    print(f"{e_70},")
    print(f"{e_75},")
    print(f"{e_80},")
    print(f"{e_85},")
    print(f"{e_90},")
    print(f"{e_95},")
    print(f"{e_99}")

    print(special_e)

    return
    
def process_pcap_mqtt(pcap_file, max_num_packets, limit=10000):
    
    camera = "10.0.0.23"
    attacker = "192.168.2.5"
    broker = "192.168.1.7"
    
    #packet_count = count_packets(pcap_file)
    
    packets = PcapReader(pcap_file)
    
    flows = {}
    
    for i, pkt in enumerate(packets, 1):
        
        #print(f"{i}/{packet_count} ({100*i/packet_count:.2f} %) - {len(flows)} flows", end="\r")
        #print(f"{i} - {len(flows)} flows", end="\r")
        
        if pkt.haslayer(Ether) and pkt.haslayer(IP):
            src_mac = pkt[Ether].src
            dst_mac = pkt[Ether].dst
    
            src_ip = pkt[IP].src
            src_port = None
            dst_ip = pkt[IP].dst
            dst_port = None
            
            if pkt.haslayer(TCP):
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif pkt.haslayer(UDP):
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
    
            if src_mac and src_ip and src_port and dst_mac and dst_ip and dst_port:
                
                flow_key = (src_ip, dst_ip)
                flow_key_rev = (dst_ip, src_ip)

                max_len = max_num_packets

                cond = src_ip == attacker or dst_ip == attacker

                if "normal" in pcap_file:
                    cond = pkt.haslayer(MQTT) or src_ip == camera or dst_ip == camera

                if "mqtt_bruteforce" in pcap_file:
                    cond = pkt.haslayer(MQTT) and (src_ip == attacker or dst_ip == attacker) and (src_ip == broker or dst_ip == broker)
                    max_len = max_num_packets*15+2
                    
                if "sparta" in pcap_file:
                    cond = pkt.haslayer(SSH) and (src_ip == attacker or dst_ip == attacker)
                    flow_key = (src_ip, dst_ip, src_port, dst_port)
                    flow_key_rev = (dst_ip, src_ip, dst_port, src_port)
                
                if cond:
                    if flow_key not in flows:
                        if flow_key_rev not in flows:
                            flows[flow_key] = []
                            flows[flow_key].append(pkt)
                        else:
                            if len(flows[flow_key_rev]) < max_len:
                                flows[flow_key_rev].append(pkt)
                    else:
                        if len(flows[flow_key]) < max_len:
                            flows[flow_key].append(pkt)
        
            if i == limit:
                if "sparta" in pcap_file:
                    break
                else:
                    if builtins.all(len(v) == max_len for v in flows.values()):
                        break
    
    return flows
    
def get_dataset_mqtt(root_dir, packet_length, max_num_packets, batch_size, num_classes, split, expand_factor, num_test_samples):
    
    pcap_file = root_dir+"normal.pcap"
    flows_benign = process_pcap_mqtt(pcap_file, max_num_packets)
    
    pcap_file = root_dir+"scan_A.pcap"
    flows_scana = process_pcap_mqtt(pcap_file, max_num_packets)
    _ = flows_scana.pop(('192.168.2.5', '10.0.0.10'))
    _ = flows_scana.pop(('192.168.2.5', '10.0.0.19'))
    
    pcap_file = root_dir+"scan_sU.pcap"
    flows_scanu = process_pcap_mqtt(pcap_file, max_num_packets)
    _ = flows_scanu.pop(('192.168.2.5', '10.0.0.10'))
    _ = flows_scanu.pop(('192.168.2.5', '10.0.0.19'))
    
    pcap_file = root_dir+"mqtt_bruteforce.pcap"
    flows_bf = process_pcap_mqtt(pcap_file, max_num_packets, limit=5000)
    del flows_bf[list(flows_bf.keys())[0]][:2]
    key, value_list = list(flows_bf.items())[0]
    chunks = [value_list[i:i + max_num_packets] for i in range(0, len(value_list), max_num_packets)]
    flows_bf = {f"{key}_{i+1}": chunk for i, chunk in enumerate(chunks)}
    
    pcap_file = root_dir+"sparta.pcap"
    flows_sparta_ = process_pcap_mqtt(pcap_file, max_num_packets, limit=5000)
    flows_sparta = {}
    for key, value in flows_sparta_.items():
        if len(value) == 12:
            flows_sparta[key] = value
        if len(flows_sparta) == 32:
            break
    
    flows_benign_ = [values for values in flows_benign.values()]
    flows_scana_ = [values for values in flows_scana.values()]
    flows_scanu_ = [values for values in flows_scanu.values()]
    flows_bf_ = [values for values in flows_bf.values()]
    flows_sparta_ = [values for values in flows_sparta.values()]

    flows_benign_ = adjust_timestamps(flows_benign_)
    flows_scana_ = adjust_timestamps(flows_scana_)
    flows_scanu_ = adjust_timestamps(flows_scanu_)
    flows_bf_ = adjust_timestamps(flows_bf_)
    flows_sparta_ = adjust_timestamps(flows_sparta_)

    selected_indices = random.sample(range(len(flows_benign_)), num_test_samples)
    flows_benign_train = [flows_benign_[i] for i in range(len(flows_benign_)) if i not in selected_indices]
    flows_benign_test = [flows_benign_[i] for i in selected_indices]

    selected_indices = random.sample(range(len(flows_scana_)), num_test_samples)
    flows_scana_train = [flows_scana_[i] for i in range(len(flows_scana_)) if i not in selected_indices]
    flows_scana_test = [flows_scana_[i] for i in selected_indices]

    selected_indices = random.sample(range(len(flows_scanu_)), num_test_samples)
    flows_scanu_train = [flows_scanu_[i] for i in range(len(flows_scanu_)) if i not in selected_indices]
    flows_scanu_test = [flows_scanu_[i] for i in selected_indices]

    selected_indices = random.sample(range(len(flows_bf_)), num_test_samples)
    flows_bf_train = [flows_bf_[i] for i in range(len(flows_bf_)) if i not in selected_indices]
    flows_bf_test = [flows_bf_[i] for i in selected_indices]

    selected_indices = random.sample(range(len(flows_sparta_)), num_test_samples)
    flows_sparta_train = [flows_sparta_[i] for i in range(len(flows_sparta_)) if i not in selected_indices]
    flows_sparta_test = [flows_sparta_[i] for i in selected_indices]

    flows_benign_sub = create_subflows(flows_benign_train)
    flows_scana_sub = create_subflows(flows_scana_train)
    flows_scanu_sub = create_subflows(flows_scanu_train)
    flows_bf_sub = create_subflows(flows_bf_train)
    flows_sparta_sub = create_subflows(flows_sparta_train)

    d_benign, t_benign, m_benign = preprocess(flows_benign_sub, packet_length, max_num_packets)
    d_scana, t_scana, m_scana = preprocess(flows_scana_sub, packet_length, max_num_packets)
    d_scanu, t_scanu, m_scanu = preprocess(flows_scanu_sub, packet_length, max_num_packets)
    d_bf, t_bf, m_bf = preprocess(flows_bf_sub, packet_length, max_num_packets)
    d_sparta, t_sparta, m_sparta = preprocess(flows_sparta_sub, packet_length, max_num_packets)

    min_number = np.min([d_benign.shape[0], d_scana.shape[0], d_scanu.shape[0], d_bf.shape[0], d_sparta.shape[0]])

    d_benign_aug, t_benign_aug, m_benign_aug = randomly_keep_elements(d_benign, t_benign, m_benign, min_number)
    d_scana_aug, t_scana_aug, m_scana_aug = randomly_keep_elements(d_scana, t_scana, m_scana, min_number)
    d_scanu_aug, t_scanu_aug, m_scanu_aug = randomly_keep_elements(d_scanu, t_scanu, m_scanu, min_number)
    d_bf_aug, t_bf_aug, m_bf_aug = randomly_keep_elements(d_bf, t_bf, m_bf, min_number)
    d_sparta_aug, t_sparta_aug, m_sparta_aug = randomly_keep_elements(d_sparta, t_sparta, m_sparta, min_number)

    d_benign_aug_ = np.tile(d_benign_aug, (expand_factor, 1, 1))
    d_scana_aug_ = np.tile(d_scana_aug, (expand_factor, 1, 1))
    d_scanu_aug_ = np.tile(d_scanu_aug, (expand_factor, 1, 1))
    d_bf_aug_ = np.tile(d_bf_aug, (expand_factor, 1, 1))
    d_sparta_aug_ = np.tile(d_sparta_aug, (expand_factor, 1, 1))

    t_benign_aug_ = np.tile(t_benign_aug, (expand_factor, 1))
    t_scana_aug_ = np.tile(t_scana_aug, (expand_factor, 1))
    t_scanu_aug_ = np.tile(t_scanu_aug, (expand_factor, 1))
    t_bf_aug_ = np.tile(t_bf_aug, (expand_factor, 1))
    t_sparta_aug_ = np.tile(t_sparta_aug, (expand_factor, 1))

    m_benign_aug_ = np.tile(m_benign_aug, (expand_factor, 1))
    m_scana_aug_ = np.tile(m_scana_aug, (expand_factor, 1))
    m_scanu_aug_ = np.tile(m_scanu_aug, (expand_factor, 1))
    m_bf_aug_ = np.tile(m_bf_aug, (expand_factor, 1))
    m_sparta_aug_ = np.tile(m_sparta_aug, (expand_factor, 1))

    x_train = np.concatenate((d_benign_aug_, d_scana_aug_, d_scanu_aug_, d_bf_aug_, d_sparta_aug_), axis=0)
    t_train = np.concatenate((t_benign_aug_, t_scana_aug_, t_scanu_aug_, t_bf_aug_, t_sparta_aug_), axis=0)
    m_train = np.concatenate((m_benign_aug_, m_scana_aug_, m_scanu_aug_, m_bf_aug_, m_sparta_aug_), axis=0)
    y = d_benign_aug_.shape[0]*[0] + d_scana_aug_.shape[0]*[1] + d_scanu_aug_.shape[0]*[2] + d_bf_aug_.shape[0]*[3] + d_sparta_aug_.shape[0]*[4]
    y_train = np.array(y)

    x_train, x_val, t_train, t_val, m_train, m_val, y_train, y_val = train_test_split(x_train, t_train, m_train, y_train, test_size=0.05, random_state=42)

    d_benign_test, t_benign_test, m_benign_test = preprocess(flows_benign_test, packet_length, max_num_packets)
    d_scana_test, t_scana_test, m_scana_test = preprocess(flows_scana_test, packet_length, max_num_packets)
    d_scanu_test, t_scanu_test, m_scanu_test = preprocess(flows_scanu_test, packet_length, max_num_packets)
    d_bf_test, t_bf_test, m_bf_test = preprocess(flows_bf_test, packet_length, max_num_packets)
    d_sparta_test, t_sparta_test, m_sparta_test = preprocess(flows_sparta_test, packet_length, max_num_packets)

    x_test = np.concatenate((d_benign_test, d_scana_test, d_scanu_test, d_bf_test, d_sparta_test), axis=0)
    t_test = np.concatenate((t_benign_test, t_scana_test, t_scanu_test, t_bf_test, t_sparta_test), axis=0)
    m_test = np.concatenate((m_benign_test, m_scana_test, m_scanu_test, m_bf_test, m_sparta_test), axis=0)
    y = d_benign_test.shape[0]*[0] + d_scana_test.shape[0]*[1] + d_scanu_test.shape[0]*[2] + d_bf_test.shape[0]*[3] + d_sparta_test.shape[0]*[4]
    y_test = np.array(y)

    print(f"- Training samples: {x_train.shape[0]}")
    print(f"- Validation samples: {x_val.shape[0]}")
    print(f"- Testing samples: {x_test.shape[0]}")

    # TF dataset
    BUFFER_SIZE = batch_size * 2
    AUTO = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices(({'flows': x_train, 'timestamps': t_train, 'masks': m_train}, y_train))
    train_ds_xm = train_ds.map(lambda inputs, labels: ({'flows': inputs['flows'], 'masks': inputs['masks']}, labels))
    train_ds_x = train_ds.map(lambda inputs, labels: ({'flows': inputs['flows']}, labels))

    train_ds = train_ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE).shuffle(BUFFER_SIZE).batch(batch_size).prefetch(AUTO)
    train_ds_xm = train_ds_xm.map(augment_data_xm, num_parallel_calls=tf.data.AUTOTUNE).shuffle(BUFFER_SIZE).batch(batch_size).prefetch(AUTO)
    train_ds_x = train_ds_x.map(augment_data_x, num_parallel_calls=tf.data.AUTOTUNE).shuffle(BUFFER_SIZE).batch(batch_size).prefetch(AUTO)

    val_ds = tf.data.Dataset.from_tensor_slices(({'flows': x_val, 'timestamps': t_val, 'masks': m_val}, y_val))
    val_ds_xm = val_ds.map(lambda inputs, labels: ({'flows': inputs['flows'], 'masks': inputs['masks']}, labels))
    val_ds_x = val_ds.map(lambda inputs, labels: ({'flows': inputs['flows']}, labels))

    val_ds = val_ds.batch(batch_size).prefetch(AUTO)
    val_ds_xm = val_ds_xm.batch(batch_size).prefetch(AUTO)
    val_ds_x = val_ds_x.batch(batch_size).prefetch(AUTO)
    
    np.savez_compressed(f"test_ds_split_{split}.npz", x_test=x_test, t_test=t_test, m_test=m_test, y_test=y_test)

    return train_ds, train_ds_xm, train_ds_x, val_ds, val_ds_xm, val_ds_x, (x_test, t_test, m_test, y_test)

def process_pcap_iotid20(pcap_file, max_num_packets, packet_count=None, limit=10000):

    victims = ["192.168.0.13", "192.168.0.24"] # camera, speaker

    # DoS time interval
    interval = 0.05 #sec
    
    packets = PcapReader(pcap_file)
    
    flows = {}

    c = 1
    flag = True
    
    for i, pkt in enumerate(packets, 1):

        #print(f"{i}/{packet_count} ({100*i/packet_count:.2f} %) - {len(flows)} flows", end="\r")
        #print(f"{i} - {len(flows)} flows", end="\r")
        
        if i == 1:
            t0 = float(pkt.time)

        if len(flows) == 600:
            break
        
        if pkt.haslayer(Ether) and pkt.haslayer(IP):
            src_mac = pkt[Ether].src
            dst_mac = pkt[Ether].dst
    
            src_ip = pkt[IP].src
            src_port = None
            dst_ip = pkt[IP].dst
            dst_port = None
            
            if pkt.haslayer(TCP):
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif pkt.haslayer(UDP):
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
    
            if src_mac and src_ip and src_port and dst_mac and dst_ip and dst_port:
                
                #flow_key = (src_ip, dst_ip)
                #flow_key_rev = (dst_ip, src_ip)

                max_len = max_num_packets

                if "benign" in pcap_file:

                    if src_ip in victims or dst_ip in victims:

                        flow_key = (src_ip, dst_ip)
                        flow_key_rev = (dst_ip, src_ip)
    
                        if flow_key not in flows:
                            if flow_key_rev not in flows:
                                flows[flow_key] = []
                                flows[flow_key].append(pkt)
                            else:
                                #if len(flows[flow_key_rev]) < max_len:
                                flows[flow_key_rev].append(pkt)
                        else:
                            #if len(flows[flow_key]) < max_len:
                            flows[flow_key].append(pkt)
                
                elif "mirai-ack" in pcap_file:
                    
                    if "192.168.0" in src_ip and "192.168.0" in dst_ip:
                        continue
                    
                    if src_ip == victims[1] or dst_ip == victims[1]:
                        if flag:
                            flow_key = f"{c}"
                            c += 1
                            flows[flow_key] = []
                            flag = False
                        if len(flows[flow_key]) < max_num_packets:
                            flows[flow_key].append(pkt)
                        else:
                            flag = True
                
                elif "mirai-http" in pcap_file:
                    
                    if "192.168.0" in src_ip and "192.168.0" in dst_ip:
                        continue
                    
                    if src_ip == victims[1] or dst_ip == victims[1]:
                        if pkt.haslayer(Raw):
                            if flag:
                                flow_key = f"{c}"
                                c += 1
                                flows[flow_key] = []
                                flag = False
                            if len(flows[flow_key]) < max_num_packets:
                                flows[flow_key].append(pkt)
                            else:
                                flag = True
                
                elif "mirai-udp" in pcap_file:
                    
                    if "192.168.0" in src_ip and "192.168.0" in dst_ip:
                        continue
                    
                    if src_ip == victims[0] or dst_ip == victims[0]:
                        if flag:
                            flow_key = f"{c}"
                            c += 1
                            flows[flow_key] = []
                            flag = False
                        if len(flows[flow_key]) < max_num_packets:
                            flows[flow_key].append(pkt)
                        else:
                            flag = True

                elif "dos" in pcap_file or "mirai" in pcap_file:
                    
                    if "192.168.0" in src_ip and "192.168.0" in dst_ip:
                        continue
                    
                    if src_ip in victims or dst_ip in victims:
                        #t = float(pkt.time)
                        if flag:
                            #t_0 = t
                            flow_key = f"{c}"
                            c += 1
                            flows[flow_key] = []
                            flag = False
                        if len(flows[flow_key]) < max_num_packets:
                            flows[flow_key].append(pkt)
                        else:
                            flag = True

                elif "mitm" in pcap_file:
                    
                    if src_ip in victims or dst_ip in victims:
                        if flag:
                            flow_key = f"{c}"
                            c += 1
                            flows[flow_key] = []
                            flag = False
                        if len(flows[flow_key]) < max_num_packets:
                            flows[flow_key].append(pkt)
                        else:
                            flag = True
                        
                elif "hostport" in pcap_file:
                    
                    attacker = "192.168.0.15"
                    
                    if (src_ip == attacker or dst_ip == attacker) and (src_ip in victims or dst_ip in victims):
                        if flag:
                            flow_key = f"{c}"
                            c += 1
                            flows[flow_key] = []
                            flag = False
                        if len(flows[flow_key]) < max_num_packets:
                            flows[flow_key].append(pkt)
                        else:
                            flag = True
                
                elif "portos" in pcap_file:
                    
                    if "portos-1" in pcap_file:
                        thres = 24
                    elif "portos-2" in pcap_file:
                        thres = 12
                    elif "portos-3" in pcap_file:
                        thres = 9
                    elif "portos-4" in pcap_file:
                        thres = 4.2

                    if float(pkt.time) - t0 < thres:
                        continue
                    
                    attacker = "192.168.0.15"
                    if (src_ip == attacker or dst_ip == attacker) and (src_ip in victims or dst_ip in victims):
                        if flag:
                            flow_key = f"{c}"
                            c += 1
                            flows[flow_key] = []
                            flag = False
                        if len(flows[flow_key]) < max_num_packets:
                            flows[flow_key].append(pkt)
                        else:
                            flag = True

    return [values for values in flows.values()]

def get_dataset_iotid20(root_dir, packet_length, max_num_packets, batch_size, num_classes, split, expand_factor, mode=1):

    # 0
    pcap_file_b = root_dir+"benign-dec.pcap"
    flows_b_ = process_pcap_iotid20(pcap_file_b, max_num_packets)
    value_list = flows_b_[0]
    flows_b = [value_list[i:i + max_num_packets] for i in range(0, len(value_list), max_num_packets)]
    flows_b = flows_b[:600]
    
    # 1
    flows_dos = []
    for n in range(1, 7):
        #print(f"n = {n}")
        pcap_file_dos = root_dir+f"dos-synflooding-{n}-dec.pcap"
        #packet_count_dos = count_packets(pcap_file_dos)
        flows_dos.extend(process_pcap_iotid20(pcap_file_dos, max_num_packets))
        #print(f"\nNum flows: {len(flows_dos)}\n")
        if len(flows_dos) >= 600:
            break

    # 2
    flows_ack = []
    for n in range(1, 2):
        #print(f"n = {n}")
        pcap_file_ack = root_dir+f"mirai-ackflooding-{n}-dec.pcap"
        #packet_count_ack = count_packets(pcap_file_ack)
        flows_ack.extend(process_pcap_iotid20(pcap_file_ack, max_num_packets))
        #print(f"\nNum flows: {len(flows_ack)}\n")
        if len(flows_ack) >= 600:
            break
    flows_ack = flows_ack[1:]
    
    # 3
    flows_bf = []
    for n in range(1, 6):
        #print(f"n = {n}")
        pcap_file_bf = root_dir+f"mirai-hostbruteforce-{n}-dec.pcap"
        #packet_count_bf = count_packets(pcap_file_bf)
        flows_bf.extend(process_pcap_iotid20(pcap_file_bf, max_num_packets))
        #print(f"\nNum flows: {len(flows_bf)}\n")
        if len(flows_bf) >= 600:
            break

    # 4
    flows_http = []
    for n in range(1, 3):
        #print(f"n = {n}")
        pcap_file_http = root_dir+f"mirai-httpflooding-{n}-dec.pcap"
        #packet_count_http = count_packets(pcap_file_http)
        flows_http.extend(process_pcap_iotid20(pcap_file_http, max_num_packets))
        #print(f"\nNum flows: {len(flows_http)}\n")
        if len(flows_http) >= 600:
            break

    # 5
    flows_udp = []
    for n in range(1, 2):
        #print(f"n = {n}")
        pcap_file_udp = root_dir+f"mirai-udpflooding-{n}-dec.pcap"
        #packet_count_udp = count_packets(pcap_file_udp)
        flows_udp.extend(process_pcap_iotid20(pcap_file_udp, max_num_packets))
        #print(f"\nNum flows: {len(flows_udp)}\n")
        if len(flows_udp) >= 600:
            break

    # 6
    flows_mitm = []
    for n in range(1, 7):
        #print(f"n = {n}")
        pcap_file_mitm = root_dir+f"mitm-arpspoofing-{n}-dec.pcap"
        #packet_count_mitm = count_packets(pcap_file_mitm)
        flows_mitm.extend(process_pcap_iotid20(pcap_file_mitm, max_num_packets))
        #print(f"\nNum flows: {len(flows_mitm)}\n")
        if len(flows_mitm) >= 600:
            break

    # 7
    flows_port = []
    for n in range(1, 7):
        #print(f"n = {n}")
        pcap_file_port = root_dir+f"scan-hostport-{n}-dec.pcap"
        #packet_count_port = count_packets(pcap_file_port)
        flows_port.extend(process_pcap_iotid20(pcap_file_port, max_num_packets))
        #print(f"\nNum flows: {len(flows_port)}\n")
        if len(flows_port) >= 600:
            break

    # 8
    flows_os = []
    for n in range(1, 5):
        #print(f"n = {n}")
        pcap_file_os = root_dir+f"scan-portos-{n}-dec.pcap"
        #packet_count_os = count_packets(pcap_file_os)
        flows_os.extend(process_pcap_iotid20(pcap_file_os, max_num_packets))
        #print(f"\nNum flows: {len(flows_os)}\n")
        if len(flows_os) >= 600:
            break

    flows_b_ = adjust_timestamps(flows_b)
    flows_dos_ = adjust_timestamps(flows_dos)
    flows_ack_ = adjust_timestamps(flows_ack)
    flows_bf_ = adjust_timestamps(flows_bf)
    flows_http_ = adjust_timestamps(flows_http)
    flows_udp_ = adjust_timestamps(flows_udp)
    flows_mitm_ = adjust_timestamps(flows_mitm)
    flows_port_ = adjust_timestamps(flows_port)
    flows_os_ = adjust_timestamps(flows_os)

    minimum = np.min([len(flows_b_), len(flows_dos_), len(flows_ack_), len(flows_bf_), len(flows_http_), len(flows_udp_), len(flows_mitm_), len(flows_port_),len(flows_os_)])
    num_test_samples = int(0.1*minimum)

    if mode == 1:
        flows_b_train = flows_b_[num_test_samples:]
        flows_b_test = flows_b_[:num_test_samples]
        flows_dos_train = flows_dos_[num_test_samples:]
        flows_dos_test = flows_dos_[:num_test_samples]
        flows_ack_train = flows_ack_[num_test_samples:]
        flows_ack_test = flows_ack_[:num_test_samples]
        flows_bf_train = flows_bf_[num_test_samples:]
        flows_bf_test = flows_bf_[:num_test_samples]
        flows_http_train = flows_http_[num_test_samples:]
        flows_http_test = flows_http_[:num_test_samples]
        flows_udp_train = flows_udp_[num_test_samples:]
        flows_udp_test = flows_udp_[:num_test_samples]
        flows_mitm_train = flows_mitm_[num_test_samples:]
        flows_mitm_test = flows_mitm_[:num_test_samples]
        flows_port_train = flows_port_[num_test_samples:]
        flows_port_test = flows_port_[:num_test_samples]
        flows_os_train = flows_os_[num_test_samples:]
        flows_os_test = flows_os_[:num_test_samples]
    elif mode == 2:
        m = num_test_samples
        n = len(flows_b_)
        flows_b_train = flows_b_[:n-m]
        flows_b_test = flows_b_[-m:]
        n = len(flows_dos_)
        flows_dos_train = flows_dos_[:n-m]
        flows_dos_test = flows_dos_[-m:]
        n = len(flows_ack_)
        flows_ack_train = flows_ack_[:n-m]
        flows_ack_test = flows_ack_[-m:]
        n = len(flows_bf_)
        flows_bf_train = flows_bf_[:n-m]
        flows_bf_test = flows_bf_[-m:]
        n = len(flows_http_)
        flows_http_train = flows_http_[:n-m]
        flows_http_test = flows_http_[-m:]
        n = len(flows_udp_)
        flows_udp_train = flows_udp_[:n-m]
        flows_udp_test = flows_udp_[-m:]
        n = len(flows_mitm_)
        flows_mitm_train = flows_mitm_[:n-m]
        flows_mitm_test = flows_mitm_[-m:]
        n = len(flows_port_)
        flows_port_train = flows_port_[:n-m]
        flows_port_test = flows_port_[-m:]
        n = len(flows_os_)
        flows_os_train = flows_os_[:n-m]
        flows_os_test = flows_os_[-m:]

    flows_b_sub = create_subflows(flows_b_train, k=2)
    flows_dos_sub = create_subflows(flows_dos_train, k=2)
    flows_ack_sub = create_subflows(flows_ack_train, k=2)
    flows_bf_sub = create_subflows(flows_bf_train, k=2)
    flows_http_sub = create_subflows(flows_http_train, k=2)
    flows_udp_sub = create_subflows(flows_udp_train, k=2)
    flows_mitm_sub = create_subflows(flows_mitm_train, k=2)
    flows_port_sub = create_subflows(flows_port_train, k=2)
    flows_os_sub = create_subflows(flows_os_train, k=2)

    d_b, t_b, m_b = preprocess(flows_b_sub, packet_length, max_num_packets)
    d_dos, t_dos, m_dos = preprocess(flows_dos_sub, packet_length, max_num_packets)
    d_ack, t_ack, m_ack = preprocess(flows_ack_sub, packet_length, max_num_packets)
    d_bf, t_bf, m_bf = preprocess(flows_bf_sub, packet_length, max_num_packets)
    d_http, t_http, m_http = preprocess(flows_http_sub, packet_length, max_num_packets)
    d_udp, t_udp, m_udp = preprocess(flows_udp_sub, packet_length, max_num_packets)
    d_mitm, t_mitm, m_mitm = preprocess(flows_mitm_sub, packet_length, max_num_packets)
    d_port, t_port, m_port = preprocess(flows_port_sub, packet_length, max_num_packets)
    d_os, t_os, m_os = preprocess(flows_os_sub, packet_length, max_num_packets)

    min_number = np.min([d_b.shape[0], d_dos.shape[0], d_ack.shape[0], d_bf.shape[0], d_http.shape[0], d_udp.shape[0], d_mitm.shape[0], d_port.shape[0], d_os.shape[0]])

    d_b_aug_, t_b_aug_, m_b_aug_ = randomly_keep_elements(d_b, t_b, m_b, min_number)
    d_dos_aug_, t_dos_aug_, m_dos_aug_ = randomly_keep_elements(d_dos, t_dos, m_dos, min_number)
    d_ack_aug_, t_ack_aug_, m_ack_aug_ = randomly_keep_elements(d_ack, t_ack, m_ack, min_number)
    d_bf_aug_, t_bf_aug_, m_bf_aug_ = randomly_keep_elements(d_bf, t_bf, m_bf, min_number)
    d_http_aug_, t_http_aug_, m_http_aug_ = randomly_keep_elements(d_http, t_http, m_http, min_number)
    d_udp_aug_, t_udp_aug_, m_udp_aug_ = randomly_keep_elements(d_udp, t_udp, m_udp, min_number)
    d_mitm_aug_, t_mitm_aug_, m_mitm_aug_ = randomly_keep_elements(d_mitm, t_mitm, m_mitm, min_number)
    d_port_aug_, t_port_aug_, m_port_aug_ = randomly_keep_elements(d_port, t_port, m_port, min_number)
    d_os_aug_, t_os_aug_, m_os_aug_ = randomly_keep_elements(d_os, t_os, m_os, min_number)

    x_train = np.concatenate((d_b_aug_, d_dos_aug_, d_ack_aug_, d_bf_aug_, d_http_aug_, d_udp_aug_, d_mitm_aug_, d_port_aug_, d_os_aug_), axis=0)
    t_train = np.concatenate((t_b_aug_, t_dos_aug_, t_ack_aug_, t_bf_aug_, t_http_aug_, t_udp_aug_, t_mitm_aug_, t_port_aug_, t_os_aug_), axis=0)
    m_train = np.concatenate((m_b_aug_, m_dos_aug_, m_ack_aug_, m_bf_aug_, m_http_aug_, m_udp_aug_, m_mitm_aug_, m_port_aug_, m_os_aug_), axis=0)
    y = d_b_aug_.shape[0]*[0] + d_dos_aug_.shape[0]*[1] + d_ack_aug_.shape[0]*[2] + d_bf_aug_.shape[0]*[3] + d_http_aug_.shape[0]*[4] + d_udp_aug_.shape[0]*[5] + d_mitm_aug_.shape[0]*[6] + d_port_aug_.shape[0]*[7] + d_os_aug_.shape[0]*[8]
    y_train = np.array(y)

    x_train, x_val, t_train, t_val, m_train, m_val, y_train, y_val = train_test_split(x_train, t_train, m_train, y_train, test_size=0.05, random_state=42)

    d_b_test, t_b_test, m_b_test = preprocess(flows_b_test, packet_length, max_num_packets)
    d_dos_test, t_dos_test, m_dos_test = preprocess(flows_dos_test, packet_length, max_num_packets)
    d_ack_test, t_ack_test, m_ack_test = preprocess(flows_ack_test, packet_length, max_num_packets)
    d_bf_test, t_bf_test, m_bf_test = preprocess(flows_bf_test, packet_length, max_num_packets)
    d_http_test, t_http_test, m_http_test = preprocess(flows_http_test, packet_length, max_num_packets)
    d_udp_test, t_udp_test, m_udp_test = preprocess(flows_udp_test, packet_length, max_num_packets)
    d_mitm_test, t_mitm_test, m_mitm_test = preprocess(flows_mitm_test, packet_length, max_num_packets)
    d_port_test, t_port_test, m_port_test = preprocess(flows_port_test, packet_length, max_num_packets)
    d_os_test, t_os_test, m_os_test = preprocess(flows_os_test, packet_length, max_num_packets)

    x_test = np.concatenate((d_b_test, d_dos_test, d_ack_test, d_bf_test, d_http_test, d_udp_test, d_mitm_test, d_port_test, d_os_test), axis=0)
    t_test = np.concatenate((t_b_test, t_dos_test, t_ack_test, t_bf_test, t_http_test, t_udp_test, t_mitm_test, t_port_test, t_os_test), axis=0)
    m_test = np.concatenate((m_b_test, m_dos_test, m_ack_test, m_bf_test, m_http_test, m_udp_test, m_mitm_test, m_port_test, m_os_test), axis=0)
    y = d_b_test.shape[0]*[0] + d_dos_test.shape[0]*[1] + d_ack_test.shape[0]*[2] + d_bf_test.shape[0]*[3] + d_http_test.shape[0]*[4] + d_udp_test.shape[0]*[5] + d_mitm_test.shape[0]*[6] + d_port_test.shape[0]*[7] + d_os_test.shape[0]*[8]
    y_test = np.array(y)

    print(f"- Training samples: {x_train.shape[0]}")
    print(f"- Validation samples: {x_val.shape[0]}")
    print(f"- Testing samples: {x_test.shape[0]}")

    # TF dataset
    BUFFER_SIZE = batch_size * 2
    AUTO = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices(({'flows': x_train, 'timestamps': t_train, 'masks': m_train}, y_train))
    train_ds_xm = train_ds.map(lambda inputs, labels: ({'flows': inputs['flows'], 'masks': inputs['masks']}, labels))
    train_ds_x = train_ds.map(lambda inputs, labels: ({'flows': inputs['flows']}, labels))

    train_ds = train_ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE).shuffle(BUFFER_SIZE).batch(batch_size).prefetch(AUTO)
    train_ds_xm = train_ds_xm.map(augment_data_xm, num_parallel_calls=tf.data.AUTOTUNE).shuffle(BUFFER_SIZE).batch(batch_size).prefetch(AUTO)
    train_ds_x = train_ds_x.map(augment_data_x, num_parallel_calls=tf.data.AUTOTUNE).shuffle(BUFFER_SIZE).batch(batch_size).prefetch(AUTO)

    val_ds = tf.data.Dataset.from_tensor_slices(({'flows': x_val, 'timestamps': t_val, 'masks': m_val}, y_val))
    val_ds_xm = val_ds.map(lambda inputs, labels: ({'flows': inputs['flows'], 'masks': inputs['masks']}, labels))
    val_ds_x = val_ds.map(lambda inputs, labels: ({'flows': inputs['flows']}, labels))

    val_ds = val_ds.batch(batch_size).prefetch(AUTO)
    val_ds_xm = val_ds_xm.batch(batch_size).prefetch(AUTO)
    val_ds_x = val_ds_x.batch(batch_size).prefetch(AUTO)

    return train_ds, train_ds_xm, train_ds_x, val_ds, val_ds_xm, val_ds_x, (x_test, t_test, m_test, y_test)


def temp():


    return







    