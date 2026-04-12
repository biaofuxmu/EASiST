import re
import torch

def compute_delays(delays_, translate_content, elapsed, src_lang, tgt_lang):
    translate_content = [content.strip() for content in translate_content]

    delays = []
    elapseds = []
    k = 0
    for i in range(len(translate_content)):
        if tgt_lang == "Chinese":
            translate_words = list(translate_content[i])
        else:
            translate_words = translate_content[i].split()
        for j in range(len(translate_words)):
            delays.append(delays_[i])
            elapseds.append(delays_[i] + elapsed[i])
            k += 1

    return delays, elapseds


def LengthAdaptiveAverageLagging(delays, source_length, target_length):
    r"""
    Length Adaptive Average Lagging (LAAL) as proposed in
    `CUNI-KIT System for Simultaneous Speech Translation Task at IWSLT 2022
    <https://arxiv.org/abs/2204.06028>`_.
    The name was suggested in `Over-Generation Cannot Be Rewarded:
    Length-Adaptive Average Lagging for Simultaneous Speech Translation
    <https://arxiv.org/abs/2206.05807>`_.
    It is the original Average Lagging as proposed in
    `Controllable Latency using Prefix-to-Prefix Framework
    <https://arxiv.org/abs/1810.08398>`_
    but is robust to the length difference between the hypothesis and reference.

    Give source :math:`X`, target :math:`Y`, delays :math:`D`,

    .. math::

        LAAL = \frac{1}{\tau} \sum_i^\tau D_i - (i - 1) \frac{|X|}{max(|Y|,|Y*|)}

    Where

    .. math::

        \tau = argmin_i(D_i = |X|)

    When reference was given, :math:`|Y|` would be the reference length, and :math:`|Y*|` is the length of the hypothesis.

    Usage:
        ----latency-metrics LAAL
    """

    if delays[0] > source_length:
        return delays[0]

    LAAL = 0
    gamma = max(len(delays), target_length) / source_length
    tau = 0
    for t_minus_1, d in enumerate(delays):
        LAAL += d - t_minus_1 / gamma
        tau = t_minus_1 + 1

        if d >= source_length:
            break
    LAAL /= tau
    return LAAL