"""
Benchmark 7 -- halting driven by the NUMBER OF REASONING STEPS, on a pretrained backbone.

Bench 6 (SST-2) showed adaptive halting saves compute, with exit depth tracking confidence.
Here the difficulty axis is something stronger and explicitly labeled: the number of
reasoning HOPS. We use bAbI qa1/qa2/qa3 -- "where is X" answerable from 1, 2, or 3 chained
supporting facts -- on a pretrained DistilBERT with the same tension early-exit head. The
question: does the operator spend MORE layers when more hops are required, with nobody telling
it the hop count?

(This is the task bench_babi.py failed on from scratch -- a flat token Transformer couldn't
even solve qa1. The pretrained backbone is the base-model competence the bench-4/5 wall said
was the precondition.)

Reuses EarlyExitTension + train + throughput from bench_glue, and the bAbI parser from
bench_babi. Story+question are tokenized as a text pair; long qa3 stories are truncated from
the LEFT (keep the most recent context + the question).

Claim under test: mean exit layer rises qa1 < qa2 < qa3, and corr(exit, #hops) > 0 -- think
longer when more reasoning steps are needed -- while matching the full model at less compute.

Run:  python3 bench_babi_pretrained.py     (expects data/ from the bAbI tar; see bench_babi.py)
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from bench_glue import EarlyExitTension, train, measure_throughput, BACKBONE, THRESHOLDS
from bench_babi import parse, BASE, FILES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAXLEN = 256


def load(dev):
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    tok.truncation_side = "left"                     # keep recent story + question
    tr = [parse(BASE + f + "_train.txt") for f in FILES.values()]
    te = [parse(BASE + f + "_test.txt") for f in FILES.values()]
    answers = sorted({a for d in tr for *_, a, _ in d})
    a2i = {a: i for i, a in enumerate(answers)}

    def enc(splits):
        stories = [" ".join(c) for d in splits for c, q, a, n in d]
        ques = [" ".join(q) for d in splits for c, q, a, n in d]
        y = [a2i[a] for d in splits for c, q, a, n in d]
        hops = [n for d in splits for c, q, a, n in d]
        b = tok(stories, ques, padding="max_length", truncation="only_first",
                max_length=MAXLEN, return_tensors="pt")
        return dict(ids=b["input_ids"].to(dev), mask=b["attention_mask"].to(dev),
                    y=torch.tensor(y, device=dev), hops=torch.tensor(hops, device=dev))
    return enc(tr), enc(te), len(answers)


@torch.no_grad()
def evaluate(model, va, dev):
    model.eval()
    logits, halts = [], []
    n = va["y"].shape[0]
    for i in range(0, n, 128):
        lg, ht = model(va["ids"][i:i+128], va["mask"][i:i+128])
        logits.append(lg); halts.append(ht)
    logits = torch.cat(logits, 1); halts = torch.sigmoid(torch.cat(halts, 1))  # (L,N,*),(L,N)
    y, hops = va["y"], va["hops"]
    L, N, _ = logits.shape

    fixedK = [(logits[k].argmax(-1) == y).float().mean().item() * 100 for k in range(L)]
    print("\nfixed-K accuracy (always exit at layer K):")
    print("  layer K :", " ".join(f"{k+1:6d}" for k in range(L)))
    print("  acc %   :", " ".join(f"{a:6.1f}" for a in fixedK))
    print(f"  full model (K={L}) = {fixedK[-1]:.2f}%")

    print("\nadaptive (tension early-exit) -- exit at first layer with halt prob >= tau:")
    print(f"{'tau':>6}{'avg layers':>11}{'acc %':>8}{'corr(exit,hops)':>16}"
          f"{'  exit qa1':>10}{'  qa2':>7}{'  qa3':>7}")
    for tau in THRESHOLDS:
        crossed = halts >= tau; crossed[-1] = True
        layer = crossed.float().argmax(0)                       # 0..L-1
        comp = layer.float() + 1
        pred = logits[layer, torch.arange(N, device=dev)].argmax(-1)
        acc = (pred == y).float().mean().item() * 100
        h = hops.float()
        corr = (((comp - comp.mean()) * (h - h.mean())).mean()
                / (comp.std() * h.std() + 1e-8)).item()
        e = {k: comp[hops == k].mean().item() for k in [1, 2, 3]}
        print(f"{tau:>6.2f}{comp.mean().item():>11.2f}{acc:>8.2f}{corr:>16.3f}"
              f"{e[1]:>10.2f}{e[2]:>7.2f}{e[3]:>7.2f}")
    print("\nread: exit layer should rise qa1<qa2<qa3 and corr(exit,hops)>0 -- more reasoning")
    print("      hops -> more layers spent, with nothing telling it the hop count.")


def main():
    dev = torch.device(DEVICE)
    torch.manual_seed(0)
    print(f"device {DEVICE}  backbone {BACKBONE}  bAbI qa1/2/3  maxlen {MAXLEN}")
    tr, te, n_cls = load(dev)
    print(f"train {tr['y'].shape[0]}  test {te['y'].shape[0]}  classes {n_cls}")
    print(f"test per-hop counts: " + ", ".join(f"qa{k}={int((te['hops']==k).sum())}" for k in [1,2,3]))

    model = EarlyExitTension(n_cls=n_cls).to(dev)
    print("training...", flush=True)
    train(model, tr, dev)
    evaluate(model, te, dev)


if __name__ == "__main__":
    main()
