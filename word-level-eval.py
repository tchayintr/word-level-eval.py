import argparse
from collections import Counter, namedtuple
from pathlib import Path
import re
import sys
'''
    ACC = # correct_w_hyp * 2 / # w_hyp -> (tp + fp) / ...
    P = (# w_ref ∩ w_hyp) / # w_hyp
    R = (# w_ref ∩ w_hyp) / # w_ref
    F1 = 2PR / (P+R)
'''

ANY_SPACE = '<SPACE>'
Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')


class Counts(object):
    def __init__(self):
        self.corrects = []  # (# w_ref ∩ w_hyp)
        self.found_corrects = []  # (# w_ref)
        self.found_guesseds = []  # (# w_hyp)
        self.found_tokens = []  # (# w_ref + w_hyp)

    def __repr__(self):
        print('correct: {}'.format(sum(self.corrects)))
        print('found_correct: {}'.format(sum(self.found_corrects)))
        print('found_guessed: {}'.format(sum(self.found_guesseds)))
        print('found_tokens: {}'.format(sum(self.found_tokens)))


def normalize_input_line(line):
    line = re.sub(' +', ' ', line).strip(' \t\n')
    return line


def test_eval(minimal=False):
    refs = ['การ เรียน หนังสือ', 'ฉัน กิน ข้าว', 'ฉัน นั่ง ตาก ลม']
    hyps = ['การ เรียน หนัง สือ', 'ฉัน กิน ข้าว', 'ฉัน นั่ง ตา กลม']
    counts = evaluate(refs, hyps)
    report(counts, minimal=minimal)


def get_n_correct(r_tokens: list, h_tokens: list) -> int:
    if len(r_tokens) > len(h_tokens):
        r_tokens, h_tokens = h_tokens, r_tokens
    r_counter = Counter(r_tokens)
    h_counter = Counter(h_tokens)
    return sum(min(h_counter[rk], rv) for rk, rv in r_counter.items())


def evaluate(refs: list, hyps: list, options: argparse.Namespace = None):
    if options is None:
        options = parse_args([])  # use default argparse

    assert len(refs) == len(hyps)
    counts = Counts()

    for ref, hyp in zip(refs, hyps):
        if options.delimiter == ANY_SPACE:
            r_tokens = ref.split()
            h_tokens = hyp.split()
        else:
            r_tokens = ref.split(options.delimiter)
            h_tokens = hyp.split(options.delimiter)

        counts.corrects.append(get_n_correct(r_tokens, h_tokens))
        counts.found_corrects.append(len(r_tokens))
        counts.found_guesseds.append(len(h_tokens))
        counts.found_tokens.append(len(r_tokens) + len(h_tokens))
    return counts


def report(counts: Counts, out=None, minimal=False):
    if out is None:
        out = sys.stdout

    overall_micro, overall_macro = metrics(counts)

    c = counts
    correct = sum(c.corrects)
    found_correct = sum(c.found_corrects)
    found_guessed = sum(c.found_guesseds)
    found_token = sum(c.found_tokens)

    if not minimal:
        out.write('processed {} tokens with {} reference tokens; '.format(
            found_token, found_correct))
        out.write('found: {} hypothesis tokens; correct: {}.\n'.format(
            found_guessed, correct))

        if found_token > 0:
            out.write('accuracy: %6.2f%%\n' %
                      (100. * correct * 2 / found_token))

            # micro
            out.write('micro: ')
            out.write('precision: %6.2f%%; ' % (100. * overall_micro.prec))
            out.write('recall: %6.2f%%; ' % (100. * overall_micro.rec))
            out.write('FB1: %6.2f\n' % (100. * overall_micro.fscore))

            # macro
            out.write('macro: ')
            out.write('precision: %6.2f%%; ' % (100. * overall_macro.prec))
            out.write('recall: %6.2f%%; ' % (100. * overall_macro.rec))
            out.write('FB1: %6.2f\n' % (100. * overall_macro.fscore))
    else:
        out.write('{}'.format(overall_micro.fscore))


def calculate_metrics(corrects, guesseds, totals, metric_type):
    if metric_type == 'micro':
        correct = sum(corrects)
        guessed = sum(guesseds)
        total = sum(totals)
        tp, fp, fn = correct, guessed - correct, total - correct
        p = 0 if tp + fp == 0 else 1. * tp / (tp + fp)
        r = 0 if tp + fn == 0 else 1. * tp / (tp + fn)
        f = 0 if p + r == 0 else 2 * p * r / (p + r)
    elif metric_type == 'macro':
        n_data = len(totals)
        mp, mr, mf = 0.0, 0.0, 0.0
        mtp, mfp, mfn = 0, 0, 0
        for correct, guessed, total in zip(corrects, guesseds, totals):
            tp, fp, fn = correct, guessed - correct, total - correct
            p = 0 if tp + fp == 0 else 1. * tp / (tp + fp)
            r = 0 if tp + fn == 0 else 1. * tp / (tp + fn)
            f = 0 if p + r == 0 else 2 * p * r / (p + r)
            mtp += tp
            mfp += fp
            mfn += fn
            mp += p
            mr += r
            mf += f
        tp = mtp / n_data
        fp = mfp / n_data
        fn = mfn / n_data
        p = mp / n_data
        r = mr / n_data
        f = mf / n_data

    return Metrics(tp, fp, fn, p, r, f)


def metrics(counts: Counts):
    c = counts
    overall_micro = calculate_metrics(c.corrects,
                                      c.found_guesseds,
                                      c.found_corrects,
                                      metric_type='micro')
    overall_macro = calculate_metrics(c.corrects,
                                      c.found_guesseds,
                                      c.found_corrects,
                                      metric_type='macro')
    return overall_micro, overall_macro


def load_data(path: Path) -> list:
    data = []
    with open(path) as f:
        for line in f:
            line = normalize_input_line(line)
            data.append(line)
    return data


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='evaluating tagging results using word-level criteria',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ref_data', type=Path, default=None)
    parser.add_argument('--hyp_data', type=Path, default=None)
    parser.add_argument('--delimiter', default=ANY_SPACE)
    parser.add_argument('--minimal',
                        action='store_true',
                        help='report only micro-f1 score')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv[1:])
    if args.test:
        counts = test_eval(minimal=args.minimal)
        sys.exit()

    if not args.ref_data or not args.hyp_data:
        print('Error: --ref_data and --hyp_data are required.',
              file=sys.stdout)
        sys.exit()

    refs = load_data(args.ref_data)
    hyps = load_data(args.hyp_data)
    counts = evaluate(refs, hyps, args)
    report(counts, minimal=args.minimal)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
