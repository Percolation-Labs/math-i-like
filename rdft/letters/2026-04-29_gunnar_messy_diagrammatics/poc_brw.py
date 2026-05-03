"""POC: BRW 1-loop diagram enumeration via rdft.crn.

Thin wrapper around ``rdft.crn.CRN.brw_thesis()`` plus the standard enumerators.
Reproduces thesis App. D's 7 distinct loop integrals + s(G)=2 for the canonical
bubble.

Run: python3 poc_brw.py
"""
from rdft.crn import CRN
from rdft.crn.enumerator import enumerate_bubbles, enumerate_tadpoles
from rdft.crn.symmetry import aut_bubble


def main():
    print("=" * 72)
    print("  BRW 1-loop enumeration via rdft.crn (thesis Eqs. 3.16+3.21)")
    print("=" * 72)

    crn = CRN.brw_thesis()
    print()
    print(f"Vertex set ({len(crn.vertices)} primitives):")
    for v in crn.vertices:
        legs = (f"in_a={v.in_dict().get('A',0)} in_b={v.in_dict().get('B',0)} "
                f"out_a={v.out_dict().get('A',0)} out_b={v.out_dict().get('B',0)}")
        print(f"  {v.name:<10}  sign={v.sign:+d}  {legs}  ({v.n_legs()}-leg)")

    bubbles = enumerate_bubbles(crn.vertices)
    tadpoles = enumerate_tadpoles(crn.vertices)
    e3 = [b for b in bubbles if sum(b.external_legs.values()) == 3]

    print()
    print(f"Enumerated:")
    print(f"  bubbles (V=2): {len(bubbles)} (E breakdown: "
          + ", ".join(f"E={E}: {sum(1 for b in bubbles if sum(b.external_legs.values())==E)}"
                      for E in (2,3,4,5,6))
          + ")")
    print(f"  tadpoles (V=1): {len(tadpoles)}")
    print(f"  total 1-loop 1PI: {len(bubbles) + len(tadpoles)}")
    print()

    # Cross-check #1: thesis Eq. (3.25) -- canonical V_branch + V_branch bubble
    s_canon = aut_bubble("V_branch", "V_branch", "A", "lr", "A", "rl")
    print(f"Cross-check thesis Eq.~(3.25): s(V_branch+V_branch, AA, opposite-dir) = {s_canon}")
    assert s_canon == 2, f"canonical bubble s changed: {s_canon}"

    # Cross-check #2: 7 distinct E=3 bubbles (thesis p.91-92)
    print(f"Cross-check thesis p.~91--92: 7 three-point bubbles  ->  POC = {len(e3)}")
    assert len(e3) == 7, f"E=3 bubble count changed: {len(e3)}"

    print()
    print("E=3 bubble panel (with s(G) and AC lineage):")
    for b in e3:
        print(f"  {b.id}")
        for prov in b.lineage[-2:]:
            print(f"    -> {prov}")
    print()
    print("All numerical checks PASSED.")


if __name__ == "__main__":
    main()
