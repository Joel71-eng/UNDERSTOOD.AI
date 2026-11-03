"""
mm_chem.py - chemistry that is computed, not painted.

Molecules and reactions are drawn with RDKit (exact bonds, exact atoms) and reactions are BALANCED by linear
algebra, so a diffusion model never has to get a bond or a coefficient right.  Process / apparatus diagrams
(distillation, electrolysis ...) go through the normal image generator instead.

    resolve_species()      name or formula -> SMILES (curated table first, optional online lookup second)
    parse_equation()       "CH4 + O2 -> CO2 + H2O" -> species + conditions
    balance()              exact integer coefficients, or None if the equation cannot be balanced
    draw_molecule()        one molecule, level-dependent style (displayed / skeletal + functional groups)
    draw_reaction()        a balanced equation as a picture with per-species regions (for the lesson animation)
"""
from __future__ import annotations

import io
import re
from collections import Counter
from fractions import Fraction
from math import gcd
from functools import reduce
from typing import Callable, Optional

from PIL import Image, ImageDraw

import mm_core as C

try:                                                                # RDKit is the only hard dependency here
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdDepictor, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    HAVE_RDKIT = True
except Exception:                                                   # noqa: BLE001
    HAVE_RDKIT = False

# =====================================================================================================
# Curated, hand-checked molecules (offline, instant).  (names..., smiles, formula label override)
# =====================================================================================================
_CURATED = [
    (["water", "h2o"], "O", "H2O"), (["hydrogen", "hydrogen gas", "dihydrogen"], "[H][H]", "H2"),
    (["oxygen", "oxygen gas", "dioxygen"], "O=O", "O2"), (["nitrogen", "nitrogen gas"], "N#N", "N2"),
    (["chlorine", "chlorine gas"], "ClCl", "Cl2"), (["carbon dioxide"], "O=C=O", "CO2"),
    (["carbon monoxide"], "[C-]#[O+]", "CO"), (["ammonia"], "N", "NH3"), (["hydrogen chloride", "hydrochloric acid"], "Cl", "HCl"),
    (["sulfur dioxide", "sulphur dioxide"], "O=S=O", "SO2"), (["sulfuric acid", "sulphuric acid"], "OS(=O)(=O)O", "H2SO4"),
    (["nitric acid"], "O[N+](=O)[O-]", "HNO3"), (["hydrogen peroxide"], "OO", "H2O2"), (["ozone"], "[O-][O+]=O", "O3"),
    (["methane"], "C", "CH4"), (["ethane"], "CC", "C2H6"), (["propane"], "CCC", "C3H8"), (["butane"], "CCCC", "C4H10"),
    (["pentane"], "CCCCC", "C5H12"), (["hexane"], "CCCCCC", "C6H14"), (["ethene", "ethylene"], "C=C", "C2H4"),
    (["propene", "propylene"], "C=CC", "C3H6"), (["ethyne", "acetylene"], "C#C", "C2H2"), (["cyclohexane"], "C1CCCCC1", "C6H12"),
    (["benzene"], "c1ccccc1", "C6H6"), (["toluene", "methylbenzene"], "Cc1ccccc1", "C7H8"), (["phenol"], "Oc1ccccc1", "C6H5OH"),
    (["methanol"], "CO", "CH3OH"), (["ethanol", "alcohol"], "CCO", "C2H5OH"), (["propan-1-ol", "propanol"], "CCCO", "C3H7OH"),
    (["ethane-1,2-diol", "ethylene glycol"], "OCCO", "C2H4(OH)2"), (["glycerol", "propane-1,2,3-triol"], "OCC(O)CO", "C3H5(OH)3"),
    (["methanal", "formaldehyde"], "C=O", "HCHO"), (["ethanal", "acetaldehyde"], "CC=O", "CH3CHO"),
    (["propanone", "acetone"], "CC(C)=O", "CH3COCH3"), (["methanoic acid", "formic acid"], "OC=O", "HCOOH"),
    (["ethanoic acid", "acetic acid", "vinegar"], "CC(=O)O", "CH3COOH"), (["ethyl ethanoate", "ethyl acetate"], "CCOC(C)=O", "CH3COOC2H5"),
    (["methylamine"], "CN", "CH3NH2"), (["ethylamine"], "CCN", "C2H5NH2"), (["chloromethane"], "CCl", "CH3Cl"),
    (["chloroethane"], "CCCl", "C2H5Cl"), (["urea"], "NC(N)=O", "CO(NH2)2"), (["glycine"], "NCC(=O)O", "C2H5NO2"),
    (["lactic acid"], "CC(O)C(=O)O", "C3H6O3"), (["citric acid"], "OC(=O)CC(O)(CC(=O)O)C(=O)O", "C6H8O7"),
    (["glucose"], "OCC1OC(O)C(O)C(O)C1O", "C6H12O6"), (["aspirin"], "CC(=O)Oc1ccccc1C(=O)O", "C9H8O4"),
    (["paracetamol"], "CC(=O)Nc1ccc(O)cc1", "C8H9NO2"), (["caffeine"], "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "C8H10N4O2"),
    (["sodium chloride", "salt", "common salt"], "[Na+].[Cl-]", "NaCl"), (["sodium hydroxide", "caustic soda"], "[Na+].[OH-]", "NaOH"),
    (["potassium hydroxide"], "[K+].[OH-]", "KOH"), (["calcium oxide", "quicklime"], "[Ca+2].[O-2]", "CaO"),
    (["magnesium oxide"], "[Mg+2].[O-2]", "MgO"), (["calcium carbonate", "limestone"], "[Ca+2].[O-]C([O-])=O", "CaCO3"),
    (["calcium hydroxide", "slaked lime"], "[Ca+2].[OH-].[OH-]", "Ca(OH)2"), (["sodium carbonate"], "[Na+].[Na+].[O-]C([O-])=O", "Na2CO3"),
    (["sodium hydrogencarbonate", "sodium bicarbonate", "baking soda"], "[Na+].OC([O-])=O", "NaHCO3"),
    (["copper(ii) sulfate", "copper sulfate", "copper(ii) sulphate"], "[Cu+2].[O-]S([O-])(=O)=O", "CuSO4"),
    (["copper(ii) oxide", "copper oxide"], "[Cu+2].[O-2]", "CuO"), (["magnesium chloride"], "[Mg+2].[Cl-].[Cl-]", "MgCl2"),
    (["calcium chloride"], "[Ca+2].[Cl-].[Cl-]", "CaCl2"), (["zinc chloride"], "[Zn+2].[Cl-].[Cl-]", "ZnCl2"),
    (["zinc sulfate", "zinc sulphate"], "[Zn+2].[O-]S([O-])(=O)=O", "ZnSO4"), (["sodium sulfate", "sodium sulphate"], "[Na+].[Na+].[O-]S([O-])(=O)=O", "Na2SO4"),
    (["iron(iii) oxide", "iron oxide"], "[Fe+3].[Fe+3].[O-2].[O-2].[O-2]", "Fe2O3"),
]
_NAME2 = {}
for _names, _smi, _lab in _CURATED:
    for _n in _names:
        _NAME2.setdefault(_n.lower(), (_smi, _lab, _names[0]))

# Reaction templates: key phrases -> (equation, conditions)
TEMPLATES = [
    (["combustion of methane", "burning methane", "methane burning"], "CH4 + O2 -> CO2 + H2O", "heat"),
    (["combustion of ethane"], "C2H6 + O2 -> CO2 + H2O", "heat"), (["combustion of propane"], "C3H8 + O2 -> CO2 + H2O", "heat"),
    (["combustion of butane"], "C4H10 + O2 -> CO2 + H2O", "heat"), (["combustion of ethanol", "burning ethanol"], "C2H5OH + O2 -> CO2 + H2O", "heat"),
    (["combustion of hydrogen", "burning hydrogen"], "H2 + O2 -> H2O", "spark"), (["burning magnesium", "combustion of magnesium"], "Mg + O2 -> MgO", "heat"),
    (["photosynthesis"], "CO2 + H2O -> C6H12O6 + O2", "light, chlorophyll"),
    (["aerobic respiration", "respiration"], "C6H12O6 + O2 -> CO2 + H2O", "enzymes"),
    (["fermentation", "alcoholic fermentation"], "C6H12O6 -> C2H5OH + CO2", "yeast, no oxygen"),
    (["neutralisation of hydrochloric acid", "neutralization of hydrochloric acid", "hcl and naoh", "hydrochloric acid and sodium hydroxide", "neutralisation", "neutralization"],
     "HCl + NaOH -> NaCl + H2O", ""),
    (["sulfuric acid and sodium hydroxide", "sulphuric acid and sodium hydroxide"], "H2SO4 + NaOH -> Na2SO4 + H2O", ""),
    (["magnesium and hydrochloric acid", "magnesium with hydrochloric acid"], "Mg + HCl -> MgCl2 + H2", ""),
    (["zinc and hydrochloric acid", "zinc with hydrochloric acid"], "Zn + HCl -> ZnCl2 + H2", ""),
    (["calcium carbonate and hydrochloric acid", "marble and hydrochloric acid"], "CaCO3 + HCl -> CaCl2 + H2O + CO2", ""),
    (["thermal decomposition of calcium carbonate", "heating calcium carbonate", "heating limestone"], "CaCO3 -> CaO + CO2", "heat"),
    (["haber process"], "N2 + H2 -> NH3", "iron catalyst, 450 C, 200 atm"), (["electrolysis of water"], "H2O -> H2 + O2", "electricity"),
    (["decomposition of hydrogen peroxide"], "H2O2 -> H2O + O2", "MnO2 catalyst"),
    (["esterification", "ethanoic acid and ethanol"], "CH3COOH + C2H5OH -> CH3COOC2H5 + H2O", "conc. H2SO4, heat"),
    (["zinc and copper sulfate", "zinc and copper sulphate", "displacement of copper"], "Zn + CuSO4 -> ZnSO4 + Cu", ""),
    (["rusting"], "Fe + O2 -> Fe2O3", "water"),
]


def known_molecule(name: str) -> bool:
    return (name or "").strip().lower() in _NAME2


# =====================================================================================================
# Formula parsing, balancing
# =====================================================================================================
def parse_formula(f: str) -> Counter:
    f = re.sub(r"\((?:s|l|g|aq)\)", "", f or "").replace("·", ".").replace("•", ".").replace("*", ".")
    total: Counter = Counter()
    for k, part in enumerate(f.split(".")):
        part = part.strip()
        mult = 1
        m = re.match(r"^(\d+)(.*)$", part)
        if m and k > 0:
            mult, part = int(m.group(1)), m.group(2)
        for el, n in _group(part).items():
            total[el] += n * mult
    if not total:
        raise ValueError(f"cannot read formula {f!r}")
    return total


def _group(s: str) -> Counter:
    stack = [Counter()]
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "([":
            stack.append(Counter())
            i += 1
        elif ch in ")]":
            j = i + 1
            while j < len(s) and s[j].isdigit():
                j += 1
            mult = int(s[i + 1:j] or 1)
            top = stack.pop()
            if not stack:
                raise ValueError("unbalanced brackets")
            for el, n in top.items():
                stack[-1][el] += n * mult
            i = j
        elif ch.isupper():
            j = i + 1
            while j < len(s) and s[j].islower():
                j += 1
            el = s[i:j]
            k = j
            while k < len(s) and s[k].isdigit():
                k += 1
            if el not in C.ELEMENTS:
                raise ValueError(f"unknown element {el!r}")
            stack[-1][el] += int(s[j:k] or 1)
            i = k
        elif ch in "+-⁺⁻" or ch.isdigit():
            i += 1                                                   # charges / stray digits are ignored for counting
        else:
            raise ValueError(f"unexpected {ch!r} in formula")
    if len(stack) != 1:
        raise ValueError("unbalanced brackets")
    return stack[0]


def _nullspace(M: list[list[Fraction]]) -> list[list[Fraction]]:
    rows, cols = len(M), len(M[0])
    A = [r[:] for r in M]
    piv, r = [], 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        pv = A[r][c]
        A[r] = [x / pv for x in A[r]]
        for i in range(rows):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
        piv.append(c)
        r += 1
        if r == rows:
            break
    basis = []
    for f in (c for c in range(cols) if c not in piv):
        v = [Fraction(0)] * cols
        v[f] = Fraction(1)
        for i, pc in enumerate(piv):
            v[pc] = -A[i][f]
        basis.append(v)
    return basis


def balance(reactants: list[str], products: list[str]) -> Optional[list[int]]:
    """Smallest positive integer coefficients (reactants first, then products), or None if not balanceable /
    ambiguous.  This is the guard that catches a wrong product suggested by the tutor."""
    try:
        counts = [parse_formula(x) for x in reactants + products]
    except ValueError:
        return None
    elems = sorted({e for c in counts for e in c})
    n_r = len(reactants)
    M = [[Fraction(counts[j].get(e, 0) * (1 if j < n_r else -1)) for j in range(len(counts))] for e in elems]
    basis = _nullspace(M)
    if len(basis) != 1:
        return None
    v = basis[0]
    if any(x == 0 for x in v) or not (all(x > 0 for x in v) or all(x < 0 for x in v)):
        return None
    if v[0] < 0:
        v = [-x for x in v]
    den = reduce(lambda a, b: a * b // gcd(a, b), [x.denominator for x in v], 1)
    ints = [int(x * den) for x in v]
    g = reduce(gcd, ints)
    return [i // g for i in ints]


def formula_markup(f: str, fmt: str = "unicode") -> str:
    """H2O -> H₂O (unicode) or H<sub>2</sub>O (reportlab)."""
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    out, i = [], 0
    while i < len(f):
        if f[i].isdigit() and i > 0 and (f[i - 1].isalpha() or f[i - 1] in ")]"):
            j = i
            while j < len(f) and f[j].isdigit():
                j += 1
            out.append(f[i:j].translate(sub) if fmt == "unicode" else f"<sub>{f[i:j]}</sub>")
            i = j
        else:
            out.append(f[i])
            i += 1
    return "".join(out)


# =====================================================================================================
# Species and equations
# =====================================================================================================
_SMILES_BY_COUNTS: dict = {}


def _counts_key(c: Counter):
    return tuple(sorted(c.items()))


def _build_index():
    if _SMILES_BY_COUNTS or not HAVE_RDKIT:
        return
    for names, smi, lab in _CURATED:
        try:
            _SMILES_BY_COUNTS.setdefault(_counts_key(parse_formula(lab)), (smi, lab, names[0]))
        except ValueError:
            pass


def resolve_species(text: str, lookup: Optional[Callable[[str], Optional[str]]] = None) -> Optional[dict]:
    """A name ('methane') or formula ('CH4') -> {label, name, smiles, counts}.  None if it cannot be understood."""
    _build_index()
    q = re.sub(r"^\d+\s*", "", (text or "").strip())
    q = re.sub(r"\((?:s|l|g|aq)\)", "", q).strip()
    if not q:
        return None
    hit = _NAME2.get(q.lower())
    if hit:
        smi, lab, nm = hit
        return {"label": lab, "name": nm, "smiles": smi, "counts": parse_formula(lab)}
    try:
        counts = parse_formula(q) if C.formula_tokens(q) or re.fullmatch(r"[A-Z][A-Za-z0-9()\[\]·.]*", q) else None
    except ValueError:
        counts = None
    if counts:
        hit = _SMILES_BY_COUNTS.get(_counts_key(counts))
        if hit:
            smi, lab, nm = hit
            return {"label": q, "name": nm, "smiles": smi, "counts": counts}
        return {"label": q, "name": None, "smiles": None, "counts": counts}          # formula only: drawn as a formula tile
    if lookup:
        try:
            smi = lookup(q)
        except Exception:                                                              # noqa: BLE001
            smi = None
        if smi and HAVE_RDKIT and Chem.MolFromSmiles(smi) is not None:
            m = Chem.MolFromSmiles(smi)
            lab = rdMolDescriptors.CalcMolFormula(m)
            return {"label": lab, "name": q, "smiles": smi, "counts": parse_formula(lab)}
    return None


ARROW = re.compile(r"\s*(?:<->|<=>|-->|->|=>|→|⟶|⇌|=)\s*")
_PREFIX = re.compile(r"^\s*(?:please\s+)?(?:balance|show|draw|write|give me|display|what is|what's)?\s*(?:the\s+)?(?:chemical\s+)?"
                     r"(?:balanced\s+)?(?:equation|reaction)?\s*(?:for|of)?\s*[:\-]?\s*", re.I)


def find_template(text: str) -> Optional[tuple[str, str]]:
    low = (text or "").lower()
    for keys, eq, cond in TEMPLATES:
        if any(k in low for k in keys):
            return eq, cond
    return None


def parse_equation(text: str) -> Optional[dict]:
    """'CH4 + O2 -> CO2 + H2O [heat]' -> {reactants, products, conditions}.  None when there is no arrow."""
    t = (text or "").strip().rstrip(".")
    cond = ""
    m = re.search(r"\[(.+?)\]", t)
    if m:
        cond, t = m.group(1).strip(), (t[:m.start()] + t[m.end():]).strip()
    parts = ARROW.split(t, maxsplit=1)
    if len(parts) != 2:
        return None
    left, right = _PREFIX.sub("", parts[0], count=1), parts[1]
    m = re.search(r"\(([A-Za-z ,.]{3,40})\)\s*$", right)
    if m and not re.fullmatch(r"(?:s|l|g|aq)", m.group(1).strip()) and not C.formula_tokens(m.group(1)):
        cond, right = cond or m.group(1).strip(), right[:m.start()].strip()

    def split(side: str) -> list[str]:
        return [re.sub(r"^\d+\s*", "", s.strip()) for s in re.split(r"\s*\+\s*(?=[0-9]*\s*[A-Za-z(])", side) if s.strip()]

    r, p = split(left), split(right)
    return {"reactants": r, "products": p, "conditions": cond} if r and p else None


def plan_reaction(text: str, lookup=None, tutor_equation: Optional[str] = None) -> dict:
    """Everything needed to draw a reaction, or {"error": ...}.  Order of trust: user's own equation, curated
    template, equation proposed by the tutor.  Whatever the source, it must BALANCE or it is rejected."""
    eq, cond, src = None, "", None
    parsed = parse_equation(text)
    if parsed:
        eq, src = parsed, "your equation"
    else:
        tpl = find_template(text)
        if tpl:
            eq, src = parse_equation(tpl[0]), "standard equation"
            if eq and tpl[1]:
                eq["conditions"] = tpl[1]
    if eq is None and tutor_equation:
        eq, src = parse_equation(tutor_equation), "suggested equation (please check it)"
    if eq is None:
        return {"error": "need_equation"}
    species = []
    for s in eq["reactants"] + eq["products"]:
        r = resolve_species(s, lookup)
        if not r:
            return {"error": f"I don't recognise '{s}' as a chemical name or formula."}
        r["query"] = s
        species.append(r)
    coeffs = balance([x["label"] for x in species[:len(eq["reactants"])]], [x["label"] for x in species[len(eq["reactants"]):]])
    if coeffs is None:
        return {"error": "That equation cannot be balanced as written - one of the substances is probably wrong or missing."}
    return {"species": species, "n_reactants": len(eq["reactants"]), "coeffs": coeffs, "conditions": eq.get("conditions", ""), "source": src}


def equation_text(plan: dict, fmt: str = "unicode") -> str:
    def side(items):
        return " + ".join((str(c) if c != 1 else "") + formula_markup(s["label"], fmt) for s, c in items)
    n = plan["n_reactants"]
    sp, co = plan["species"], plan["coeffs"]
    arrow = "→" if fmt == "unicode" else "->"
    return f"{side(list(zip(sp[:n], co[:n])))} {arrow} {side(list(zip(sp[n:], co[n:])))}"


# =====================================================================================================
# Drawing
# =====================================================================================================
GROUPS = [  # (name, SMARTS)
    ("carboxylic acid", "[CX3](=O)[OX2H1]"), ("ester", "[#6][CX3](=O)[OX2H0][#6]"), ("amide", "[CX3](=O)[NX3]"),
    ("aldehyde", "[CX3H1](=O)[#6,#1]"), ("ketone", "[#6][CX3](=O)[#6]"), ("alcohol", "[CX4][OX2H]"), ("phenol", "c[OX2H]"),
    ("ether", "[OD2]([#6])[#6]"), ("amine", "[NX3;H2,H1;!$(NC=O)]"), ("alkene", "C=C"), ("alkyne", "C#C"),
    ("halogenoalkane", "[CX4][F,Cl,Br,I]"), ("benzene ring", "c1ccccc1"),
]


def find_groups(mol) -> list[dict]:
    out, taken = [], set()
    for name, smarts in GROUPS:
        patt = Chem.MolFromSmarts(smarts)
        for match in mol.GetSubstructMatches(patt):
            if set(match) & taken and name not in ("benzene ring",):
                continue
            out.append({"name": name, "atoms": list(match)})
            taken |= set(match)
            break
    return out


def _png_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def mol_image(smiles: str, size: int = 520, style: str = "skeletal", highlight: Optional[list] = None, stereo: bool = False) -> Image.Image:
    """highlight = [(atom_indices, (r,g,b) 0-255), ...]"""
    if not HAVE_RDKIT:
        raise RuntimeError("RDKit is not installed (pip install rdkit)")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"invalid structure {smiles!r}")
    heavy = m.GetNumHeavyAtoms()
    ionic = any(a.GetFormalCharge() != 0 for a in m.GetAtoms()) and "." in smiles
    if style == "displayed" and heavy <= 14 and not ionic:
        m = Chem.AddHs(m)
    rdDepictor.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DCairo(size, int(size * 0.8))
    o = d.drawOptions()
    o.bondLineWidth = 3
    o.addStereoAnnotation = bool(stereo)
    o.padding = 0.12
    atoms, colours = [], {}
    for idx_list, col in (highlight or []):
        for i in idx_list:
            atoms.append(i)
            colours[i] = tuple(c / 255 for c in col)
    d.DrawMolecule(m, highlightAtoms=atoms or None, highlightAtomColors=colours or None)
    d.FinishDrawing()
    return _png_to_pil(d.GetDrawingText())


def draw_molecule(species: dict, level_key: str) -> dict:
    """Returns {image, groups:[{name, colour, image}], facts}.  Style depends on the study level."""
    lv = C.get_level(level_key)
    style = lv["mol_style"]
    smi = species["smiles"]
    m = Chem.MolFromSmiles(smi)
    groups = find_groups(m) if style != "displayed" else []
    for i, g in enumerate(groups):
        g["colour"] = C.colour_for(i)
    stereo = style == "skeletal_full"
    base = mol_image(smi, 560, "displayed" if style == "displayed" else "skeletal",
                     [(g["atoms"], g["colour"]) for g in groups] if style != "displayed" else None, stereo)
    for g in groups:
        g["image"] = mol_image(smi, 560, "skeletal", [(g["atoms"], g["colour"])], stereo)
    facts = {"formula": species["label"], "name": species.get("name"), "molar_mass": round(Descriptors.MolWt(m), 1),
             "heavy_atoms": m.GetNumHeavyAtoms(), "smiles": smi}
    return {"image": base, "groups": groups, "facts": facts}


def _draw_formula(d: ImageDraw.ImageDraw, xy, formula: str, font, small, fill=(30, 30, 30)):
    x, y = xy
    i = 0
    while i < len(formula):
        ch = formula[i]
        if ch.isdigit() and i > 0 and (formula[i - 1].isalpha() or formula[i - 1] in ")]"):
            d.text((x, y + font.size * 0.35), ch, font=small, fill=fill)
            x += d.textlength(ch, font=small)
        else:
            d.text((x, y), ch, font=font, fill=fill)
            x += d.textlength(ch, font=font)
        i += 1
    return x


def _formula_width(d, formula, font, small) -> float:
    w = 0.0
    for i, ch in enumerate(formula):
        sub = ch.isdigit() and i > 0 and (formula[i - 1].isalpha() or formula[i - 1] in ")]")
        w += d.textlength(ch, font=small if sub else font)
    return w


def formula_tile(label: str, size: int) -> Image.Image:
    img = Image.new("RGB", (size, int(size * 0.8)), "white")
    d = ImageDraw.Draw(img)
    f, s = C.get_font(int(size * 0.2), True), C.get_font(int(size * 0.13), True)
    w = _formula_width(d, label, f, s)
    _draw_formula(d, ((size - w) / 2, size * 0.28), label, f, s)
    return img


def draw_reaction(plan: dict, level_key: str) -> dict:
    """Balanced equation as one picture.  Returns {image, regions:[{name, bbox, colour}], equation, conditions}."""
    lv = C.get_level(level_key)
    sp, co, n = plan["species"], plan["coeffs"], plan["n_reactants"]
    tile = 300 if len(sp) <= 4 else 250
    th = int(tile * 0.8)
    name_font, big, small = C.get_font(int(tile * 0.095)), C.get_font(int(tile * 0.17), True), C.get_font(int(tile * 0.11), True)
    cfont = C.get_font(int(tile * 0.22), True)
    tiles = []
    for s in sp:
        try:
            im = mol_image(s["smiles"], tile, "displayed" if lv["mol_style"] == "displayed" else "skeletal") if s.get("smiles") else formula_tile(s["label"], tile)
        except Exception:                                           # noqa: BLE001
            im = formula_tile(s["label"], tile)
        tiles.append(im.resize((tile, th)))
    label_h = int(tile * 0.38)
    sign_w, arrow_w = int(tile * 0.22), int(tile * 0.95)
    coef_w = int(tile * 0.2)
    total_w = 20
    seq = []
    for i, s in enumerate(sp):
        if i == n:
            seq.append(("arrow", arrow_w))
        elif i > 0:
            seq.append(("plus", sign_w))
        seq.append(("tile", i, coef_w + tile))
    for it in seq:
        total_w += it[-1] if it[0] == "tile" else it[1]
    total_w += 20
    H = th + label_h + int(tile * 0.35)
    canvas = Image.new("RGB", (total_w, H), "white")
    d = ImageDraw.Draw(canvas)
    x, regions = 20, []
    mid = int(tile * 0.2) + th // 2
    for it in seq:
        if it[0] == "plus":
            d.text((x + sign_w / 2, mid), "+", font=cfont, fill=(60, 60, 60), anchor="mm")
            x += sign_w
        elif it[0] == "arrow":
            y = mid
            C._arrow(d, (x + 8, y), (x + arrow_w - 8, y), (60, 60, 60), 4)
            cond = plan.get("conditions") or ""
            if cond:
                d.text((x + arrow_w / 2, y - int(tile * 0.13)), cond, font=name_font, fill=(60, 60, 60), anchor="mm")
            x += arrow_w
        else:
            i = it[1]
            if co[i] != 1:
                d.text((x + coef_w / 2, mid), str(co[i]), font=cfont, fill=(30, 30, 30), anchor="mm")
            tx = x + coef_w
            canvas.paste(tiles[i], (tx, int(tile * 0.2)))
            lab = sp[i]["label"]
            w = _formula_width(d, lab, big, small)
            _draw_formula(d, (tx + (tile - w) / 2, int(tile * 0.2) + th + 4), lab, big, small)
            nm = sp[i].get("name")
            if nm:
                d.text((tx + tile / 2, int(tile * 0.2) + th + int(tile * 0.31)), nm, font=name_font, fill=(100, 100, 100), anchor="mm")
            regions.append({"name": nm or lab, "label": lab, "bbox": (tx - coef_w // 2, int(tile * 0.1), tx + tile, H - 4),
                            "colour": C.colour_for(i), "role": "reactant" if i < n else "product", "coeff": co[i]})
            x += coef_w + tile
    return {"image": canvas, "regions": regions, "equation": equation_text(plan), "conditions": plan.get("conditions", "")}


def atom_balance(plan: dict) -> list[dict]:
    """Per-element atom counts on each side (for the explanation and the PDF)."""
    n, sp, co = plan["n_reactants"], plan["species"], plan["coeffs"]
    left, right = Counter(), Counter()
    for i, s in enumerate(sp):
        for el, k in s["counts"].items():
            (left if i < n else right)[el] += k * co[i]
    return [{"element": e, "left": left[e], "right": right[e]} for e in sorted(set(left) | set(right))]


GROUP_NOTES = {
    "alcohol": "The OH group is the alcohol functional group. It lets the molecule form hydrogen bonds, so small alcohols dissolve in water.",
    "carboxylic acid": "The COOH group is the carboxylic acid functional group. It can give away a hydrogen ion, which makes the molecule acidic.",
    "ester": "The ester group joins an acid part and an alcohol part. Esters often have sweet, fruity smells.",
    "amide": "The amide group links carbon and nitrogen and is the bond that joins amino acids in proteins.",
    "aldehyde": "The aldehyde group has a carbon double bonded to oxygen at the end of the chain.",
    "ketone": "The ketone group has a carbon double bonded to oxygen inside the chain.",
    "phenol": "A hydroxyl group attached directly to a benzene ring is a phenol.",
    "ether": "The ether group is an oxygen atom between two carbon atoms.",
    "amine": "The amine group contains nitrogen and can act as a base.",
    "alkene": "The carbon to carbon double bond makes the molecule unsaturated and more reactive.",
    "alkyne": "The carbon to carbon triple bond makes the molecule unsaturated.",
    "halogenoalkane": "A halogen atom is attached to a carbon chain.",
    "benzene ring": "The benzene ring is a stable ring of six carbon atoms with delocalised electrons.",
}
