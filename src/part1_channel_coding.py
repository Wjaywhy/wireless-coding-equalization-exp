"""
Part 1：信道编码实验

学生需要完成 Hamming(7,4) 编码、伴随式计算和单比特纠错译码。
选做内容包括卷积码编码和 Viterbi 硬判决译码。
"""

import numpy as np
from utils import (
    binary_symmetric_channel,
    calculate_ber,
    generate_bits,
    plot_ber_curve,
)

HAMMING_G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=int)

HAMMING_H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=int)


def hamming74_encode(bits):
    """
    Hamming(7,4) 系统码编码。

    参数:
        bits: 一维 0/1 数组，长度必须是 4 的倍数。

    返回:
        encoded: 一维 0/1 编码比特数组，长度为输入的 7/4 倍。

    要求:
        使用课件中的生成矩阵 G，按 GF(2) 进行矩阵乘法。
    """
    bits = np.asarray(bits, dtype=int)
    if bits.ndim != 1:
        raise ValueError('bits 必须是一维数组')
    if len(bits) % 4 != 0:
        raise ValueError('Hamming(7,4) 要求输入长度为 4 的倍数')
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits 只能包含 0 或 1')

    blocks = bits.reshape(-1, 4)
    encoded = (blocks @ HAMMING_G) % 2
    return encoded.reshape(-1)


def hamming74_syndrome(codewords):
    """
    计算 Hamming(7,4) 码字的伴随式。

    参数:
        codewords: 一维或二维 0/1 数组。若为一维，长度必须是 7 的倍数。

    返回:
        syndromes: 形状为 (N, 3) 的伴随式数组。
    """
    codewords = np.asarray(codewords, dtype=int)
    if codewords.ndim == 1:
        if len(codewords) % 7 != 0:
            raise ValueError('码字长度必须是 7 的倍数')
        codewords = codewords.reshape(-1, 7)
    if codewords.shape[1] != 7:
        raise ValueError('每个 Hamming(7,4) 码字长度必须为 7')

    syndromes = (codewords @ HAMMING_H.T) % 2
    return syndromes


def hamming74_decode(received):
    """
    Hamming(7,4) 单比特纠错译码。

    参数:
        received: 一维 0/1 接收序列，长度必须是 7 的倍数。

    返回:
        decoded_bits: 纠错后提取出的信息比特序列。

    提示:
        1. 计算每个码字的伴随式。
        2. 若伴随式非零，将其与 H 的各列比较，定位错误比特。
        3. 翻转对应错误位。
        4. 系统码的信息位为前 4 位。
    """
    received = np.asarray(received, dtype=int)
    if received.ndim != 1 or len(received) % 7 != 0:
        raise ValueError('received 必须是一维数组，长度为 7 的倍数')

    codewords = received.reshape(-1, 7).copy()
    syndromes = hamming74_syndrome(codewords)

    # 预计算伴随式到错误位置的映射
    syndrome_map = {tuple(HAMMING_H[:, i] % 2): i for i in range(7)}

    for idx, syndrome in enumerate(syndromes):
        if np.any(syndrome):
            error_pos = syndrome_map.get(tuple(syndrome))
            if error_pos is not None:
                codewords[idx, error_pos] ^= 1

    decoded_bits = codewords[:, :4].reshape(-1)
    return decoded_bits


def convolutional_encode(bits):
    """
    选做：实现 (2,1,3) 卷积码编码，生成多项式为 g1=111, g2=101。

    默认在末尾添加 2 个 0 作为尾比特，使状态回到全零。
    """
    bits = np.asarray(bits, dtype=int)
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits 只能包含 0 或 1')

    tail = np.zeros(2, dtype=int)
    bits = np.concatenate([bits, tail])
    state = [0, 0]  # s1, s2
    outputs = []
    for bit in bits:
        s1, s2 = state
        g1 = bit ^ s1 ^ s2
        g2 = bit ^ s2
        outputs.extend([g1, g2])
        state = [bit, s1]
    return np.array(outputs, dtype=int)


def viterbi_decode_hard(received_bits):
    """
    选做：实现 (2,1,3) 卷积码硬判决 Viterbi 译码。
    """
    received_bits = np.asarray(received_bits, dtype=int)
    if len(received_bits) % 2 != 0:
        raise ValueError('卷积码接收序列长度必须是 2 的倍数')

    num_symbols = len(received_bits) // 2
    received_pairs = received_bits.reshape(-1, 2)

    num_states = 4  # memory=2
    inf = 1e9
    path_metrics = np.full(num_states, inf, dtype=float)
    path_metrics[0] = 0.0
    prev_state = np.full((num_symbols, num_states), -1, dtype=int)
    prev_bit = np.full((num_symbols, num_states), -1, dtype=int)

    # State representation: (s1, s2) -> s1*2 + s2
    for t in range(num_symbols):
        rx = received_pairs[t]
        next_metrics = np.full(num_states, inf, dtype=float)
        next_prev_state = np.full(num_states, -1, dtype=int)
        next_prev_bit = np.full(num_states, -1, dtype=int)

        for state in range(num_states):
            if path_metrics[state] >= inf:
                continue
            s1 = (state >> 1) & 1
            s2 = state & 1
            for bit in (0, 1):
                g1 = bit ^ s1 ^ s2
                g2 = bit ^ s2
                dist = (g1 != rx[0]) + (g2 != rx[1])
                next_state = (bit << 1) | s1
                metric = path_metrics[state] + dist
                if metric < next_metrics[next_state]:
                    next_metrics[next_state] = metric
                    next_prev_state[next_state] = state
                    next_prev_bit[next_state] = bit

        path_metrics = next_metrics
        prev_state[t] = next_prev_state
        prev_bit[t] = next_prev_bit

    # Prefer terminating to all-zero state when tail bits are used
    final_state = 0
    if path_metrics[final_state] >= inf:
        final_state = int(np.argmin(path_metrics))

    decoded = np.zeros(num_symbols, dtype=int)
    state = final_state
    for t in range(num_symbols - 1, -1, -1):
        bit = prev_bit[t, state]
        decoded[t] = 0 if bit < 0 else bit
        state = prev_state[t, state] if prev_state[t, state] >= 0 else 0

    if num_symbols >= 2:
        decoded = decoded[:-2]
    return decoded


def run_coding_demo():
    """运行 Part 1 演示并生成 BER 曲线。"""
    print('=' * 60)
    print('Part 1：信道编码实验')
    print('=' * 60)

    error_probabilities = np.array([0.001, 0.003, 0.01, 0.03, 0.06, 0.1])
    uncoded_ber = []
    coded_ber = []

    try:
        bits = generate_bits(4000, seed=2026)
        bits = bits[: len(bits) // 4 * 4]
        encoded = hamming74_encode(bits)

        for index, probability in enumerate(error_probabilities):
            uncoded_rx = binary_symmetric_channel(bits, probability, seed=100 + index)
            encoded_rx = binary_symmetric_channel(encoded, probability, seed=200 + index)
            decoded = hamming74_decode(encoded_rx)
            uncoded_ber.append(calculate_ber(bits, uncoded_rx))
            coded_ber.append(calculate_ber(bits, decoded))

        plot_ber_curve(
            error_probabilities,
            {'未编码': uncoded_ber, 'Hamming(7,4)': coded_ber},
            'Hamming(7,4) 编码前后 BER 对比',
            'coding_ber_curve.png',
        )
        print('✅ 已生成 results/coding_ber_curve.png')
    except NotImplementedError as error:
        print(f'⏸️ 尚未完成核心函数：{error}')
    except Exception as error:
        print(f'❌ Part 1 运行失败：{error}')


if __name__ == '__main__':
    run_coding_demo()
