"""
mm_sources.py - where the facts come from (runs on the PC, cached in SQLite so a repeat topic works offline).

Order of trust for a diagram topic
    1. Built-in notes  - hand-written, vetted lists of the standard labelled parts (instant, offline)
    2. Wikipedia       - fetched, cached, attributed (CC BY-SA); the tutor reads it and proposes parts,
                         code then keeps only names that literally appear in the text
Adding a new source later (Gene Ontology, OpenStax ...) = one more function that returns the same dict.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

import requests

import mm_core as C

UA = "UnderstoodCapstone/2.0 (student project, MUST Uganda)"


class SourceStore:
    """Tiny key-value cache: a topic fetched once is available offline."""

    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute("CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, value TEXT, ts REAL)")
            self.conn.commit()

    def get(self, key: str):
        with self.lock:
            row = self.conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value):
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO cache VALUES(?,?,?)", (key, json.dumps(value), time.time()))
            self.conn.commit()


# =====================================================================================================
# Built-in notes.  Each topic: ordered parts (most important first) with ONE accurate sentence each.
# The level decides how many are used (Ordinary 6, Advanced 9, University 13).
# =====================================================================================================
def _t(aliases, intro, parts, subject="biology"):
    return {"aliases": aliases, "intro": intro, "parts": parts, "subject": subject}


CURATED: dict[str, dict] = {
    "plant cell": _t(["plant cells", "plant cell structure"], "A plant cell is the basic unit of a plant and has structures that animal cells lack.", [
        ("cell wall", "The cell wall is a rigid layer of cellulose outside the cell membrane that supports the cell and gives it shape."),
        ("cell membrane", "The cell membrane controls which substances enter and leave the cell."),
        ("nucleus", "The nucleus contains the DNA and controls the activities of the cell."),
        ("cytoplasm", "The cytoplasm is the jelly-like material where many chemical reactions take place."),
        ("chloroplast", "Chloroplasts contain chlorophyll and are where photosynthesis happens."),
        ("vacuole", "The large vacuole holds cell sap and keeps the cell firm."),
        ("mitochondrion", "Mitochondria release energy from glucose during aerobic respiration."),
        ("ribosome", "Ribosomes make proteins."),
        ("endoplasmic reticulum", "The endoplasmic reticulum is a network of membranes that transports and helps make proteins and lipids."),
        ("golgi apparatus", "The Golgi apparatus modifies, sorts and packages proteins."),
        ("nucleolus", "The nucleolus inside the nucleus makes the parts of ribosomes."),
        ("plasmodesmata", "Plasmodesmata are narrow channels through the cell wall that connect neighbouring plant cells."),
        ("starch grain", "Starch grains are stores of carbohydrate inside the cell."),
    ]),
    "animal cell": _t(["animal cells", "animal cell structure"], "An animal cell is the basic unit of an animal's body and has no cell wall or chloroplasts.", [
        ("cell membrane", "The cell membrane controls which substances enter and leave the cell."),
        ("nucleus", "The nucleus contains the DNA and controls the activities of the cell."),
        ("cytoplasm", "The cytoplasm is the jelly-like material where many chemical reactions take place."),
        ("mitochondrion", "Mitochondria release energy from glucose during aerobic respiration."),
        ("ribosome", "Ribosomes make proteins."),
        ("vacuole", "Animal cells have small vacuoles that store substances temporarily."),
        ("endoplasmic reticulum", "The endoplasmic reticulum is a network of membranes that transports and helps make proteins and lipids."),
        ("golgi apparatus", "The Golgi apparatus modifies, sorts and packages proteins."),
        ("lysosome", "Lysosomes contain enzymes that digest waste materials and worn-out parts of the cell."),
        ("nucleolus", "The nucleolus inside the nucleus makes the parts of ribosomes."),
        ("centrosome", "The centrosome organises the microtubules used when the cell divides."),
    ]),
    "human heart": _t(["heart", "the heart", "human heart anatomy", "mammalian heart"], "The heart is a muscular pump with four chambers that pumps blood around the body.", [
        ("right atrium", "The right atrium receives deoxygenated blood from the body through the vena cava."),
        ("right ventricle", "The right ventricle pumps deoxygenated blood to the lungs through the pulmonary artery."),
        ("left atrium", "The left atrium receives oxygenated blood from the lungs through the pulmonary vein."),
        ("left ventricle", "The left ventricle has a thick muscular wall and pumps oxygenated blood to the body through the aorta."),
        ("aorta", "The aorta is the largest artery and carries oxygenated blood from the left ventricle to the body."),
        ("vena cava", "The vena cava brings deoxygenated blood from the body back to the right atrium."),
        ("pulmonary artery", "The pulmonary artery carries deoxygenated blood from the right ventricle to the lungs."),
        ("pulmonary vein", "The pulmonary vein carries oxygenated blood from the lungs to the left atrium."),
        ("septum", "The septum is the wall that separates the left and right sides of the heart."),
        ("tricuspid valve", "The tricuspid valve stops blood flowing back from the right ventricle to the right atrium."),
        ("bicuspid valve", "The bicuspid valve stops blood flowing back from the left ventricle to the left atrium."),
        ("coronary artery", "Coronary arteries supply the heart muscle itself with oxygenated blood."),
    ]),
    "neuron": _t(["nerve cell", "neurone", "motor neuron", "motor neurone"], "A neuron is a nerve cell that carries electrical impulses.", [
        ("cell body", "The cell body contains the nucleus and keeps the neuron alive."),
        ("dendrites", "Dendrites receive signals from other neurons and carry them to the cell body."),
        ("axon", "The axon is a long fibre that carries the impulse away from the cell body."),
        ("myelin sheath", "The myelin sheath is a fatty layer around the axon that insulates it and speeds up the impulse."),
        ("axon terminals", "The axon terminals pass the signal on to the next cell across a synapse."),
        ("nucleus", "The nucleus contains the DNA of the neuron."),
        ("node of Ranvier", "Nodes of Ranvier are gaps in the myelin sheath where the impulse jumps along the axon."),
        ("Schwann cell", "Schwann cells make the myelin sheath around the axon."),
    ]),
    "human brain": _t(["brain", "the brain", "human brain anatomy", "parts of the brain", "brain structure"], "The brain is the control centre of the nervous system, inside the skull.", [
        ("cerebrum", "The cerebrum is the large folded upper part of the brain used for thinking, memory, speech and voluntary movement."),
        ("cerebellum", "The cerebellum at the back of the brain coordinates movement and keeps the body balanced."),
        ("brain stem", "The brain stem joins the brain to the spinal cord and controls breathing and heart rate."),
        ("corpus callosum", "The corpus callosum is a thick band of nerve fibres that joins the left and right halves of the cerebrum."),
        ("thalamus", "The thalamus relays sensory signals to the correct part of the cerebrum."),
        ("hypothalamus", "The hypothalamus controls body temperature, hunger, thirst and hormone release."),
        ("pituitary gland", "The pituitary gland hangs below the hypothalamus and releases hormones."),
        ("hippocampus", "The hippocampus helps to form new memories."),
        ("amygdala", "The amygdala processes emotions such as fear."),
    ]),
    "reflex arc": _t(["reflex action", "reflex pathway", "reflex arc pathway"], "A reflex arc is the fast, automatic pathway from a stimulus to a response.", [
        ("receptor", "The receptor detects the stimulus, for example heat or pain, and starts an impulse."),
        ("sensory neuron", "The sensory neuron carries the impulse from the receptor to the spinal cord."),
        ("relay neuron", "The relay neuron in the spinal cord passes the impulse from the sensory neuron to the motor neuron."),
        ("motor neuron", "The motor neuron carries the impulse from the spinal cord to the effector."),
        ("effector", "The effector, a muscle or gland, produces the response."),
        ("spinal cord", "The spinal cord is the coordinator of the reflex."),
    ]),
    "nephron": _t(["kidney nephron", "kidney tubule", "renal nephron"], "The nephron is the filtering unit of the kidney.", [
        ("glomerulus", "The glomerulus is a knot of capillaries where blood is filtered under high pressure."),
        ("Bowman's capsule", "Bowman's capsule surrounds the glomerulus and collects the filtrate."),
        ("proximal convoluted tubule", "The proximal convoluted tubule reabsorbs glucose, amino acids and most of the water and salts."),
        ("loop of Henle", "The loop of Henle creates a salt gradient that allows water to be reabsorbed."),
        ("distal convoluted tubule", "The distal convoluted tubule adjusts the amount of salts and water reabsorbed."),
        ("collecting duct", "The collecting duct carries urine to the renal pelvis and reabsorbs more water when needed."),
        ("afferent arteriole", "The afferent arteriole brings blood to the glomerulus."),
        ("efferent arteriole", "The efferent arteriole takes blood away from the glomerulus."),
        ("peritubular capillaries", "Peritubular capillaries surround the tubules and take up the reabsorbed substances."),
    ]),
    "flower": _t(["flower structure", "parts of a flower", "insect pollinated flower"], "A flower is the reproductive organ of a flowering plant.", [
        ("petal", "Petals are often brightly coloured to attract insects for pollination."),
        ("sepal", "Sepals protect the flower while it is a bud."),
        ("stamen", "The stamen is the male part of the flower and makes pollen."),
        ("anther", "The anther is the part of the stamen that produces pollen grains."),
        ("filament", "The filament is the stalk that holds up the anther."),
        ("carpel", "The carpel is the female part of the flower."),
        ("stigma", "The stigma is the sticky tip of the carpel that catches pollen."),
        ("style", "The style is the stalk that connects the stigma to the ovary."),
        ("ovary", "The ovary contains the ovules and becomes the fruit after fertilisation."),
        ("ovule", "An ovule contains the female sex cell and becomes the seed after fertilisation."),
        ("receptacle", "The receptacle is the base of the flower that holds all the parts."),
    ]),
    "leaf": _t(["leaf structure", "leaf cross section", "cross section of a leaf", "plant leaf"], "A leaf is the main organ of photosynthesis in a plant.", [
        ("upper epidermis", "The upper epidermis is a thin transparent layer that protects the leaf and lets light through."),
        ("palisade mesophyll", "The palisade mesophyll cells contain many chloroplasts and carry out most photosynthesis."),
        ("spongy mesophyll", "The spongy mesophyll has air spaces that allow gases to move through the leaf."),
        ("lower epidermis", "The lower epidermis contains most of the stomata."),
        ("stoma", "A stoma is a pore that allows carbon dioxide in and oxygen and water vapour out."),
        ("guard cell", "Guard cells change shape to open and close the stoma."),
        ("vein", "A vein contains xylem and phloem that carry water and sugars."),
        ("cuticle", "The waxy cuticle reduces water loss from the leaf."),
        ("xylem", "Xylem vessels carry water and minerals up to the leaf."),
        ("phloem", "Phloem tubes carry sugars away from the leaf."),
    ]),
    "bacterial cell": _t(["bacterium", "bacteria cell", "prokaryotic cell", "bacteria"], "A bacterial cell is a simple cell with no true nucleus.", [
        ("cell wall", "The cell wall gives the bacterium its shape and protects it."),
        ("cell membrane", "The cell membrane controls what enters and leaves the cell."),
        ("cytoplasm", "The cytoplasm is where the chemical reactions of the cell take place."),
        ("nucleoid", "The nucleoid is the region where the single loop of DNA is found."),
        ("plasmid", "A plasmid is a small ring of extra DNA that can carry genes such as antibiotic resistance."),
        ("ribosome", "Ribosomes make proteins."),
        ("flagellum", "The flagellum is a whip-like tail that moves the cell."),
        ("capsule", "The capsule is a slimy layer that protects the cell."),
        ("pilus", "Pili are short hairs that help the cell attach to surfaces."),
    ]),
    "human eye": _t(["eye", "the eye", "eye structure", "parts of the eye"], "The eye detects light and sends signals to the brain.", [
        ("cornea", "The cornea is the clear front surface that refracts light into the eye."),
        ("iris", "The iris controls how much light enters by changing the size of the pupil."),
        ("pupil", "The pupil is the opening in the iris that lets light in."),
        ("lens", "The lens focuses light onto the retina."),
        ("retina", "The retina contains light-sensitive cells that detect light."),
        ("optic nerve", "The optic nerve carries impulses from the retina to the brain."),
        ("sclera", "The sclera is the tough white outer layer that protects the eye."),
        ("ciliary muscle", "Ciliary muscles change the shape of the lens to focus on near or distant objects."),
        ("vitreous humour", "The vitreous humour is the clear jelly that fills the eyeball and keeps its shape."),
        ("choroid", "The choroid is a layer rich in blood vessels that supplies the retina."),
    ]),
    "digestive system": _t(["human digestive system", "alimentary canal", "digestion system"], "The digestive system breaks food down and absorbs nutrients.", [
        ("mouth", "In the mouth, teeth break up food and saliva starts the digestion of starch."),
        ("oesophagus", "The oesophagus carries food to the stomach by peristalsis."),
        ("stomach", "The stomach churns food and begins protein digestion with acid and enzymes."),
        ("liver", "The liver makes bile, which helps digest fats."),
        ("gall bladder", "The gall bladder stores bile."),
        ("pancreas", "The pancreas makes digestive enzymes and insulin."),
        ("small intestine", "The small intestine completes digestion and absorbs digested food into the blood."),
        ("large intestine", "The large intestine absorbs water from the undigested food."),
        ("rectum", "The rectum stores faeces before they leave the body."),
        ("anus", "The anus is the opening through which faeces leave the body."),
    ]),
    "respiratory system": _t(["human respiratory system", "lungs", "breathing system", "gas exchange system"], "The respiratory system brings oxygen into the body and removes carbon dioxide.", [
        ("trachea", "The trachea is the windpipe that carries air to the lungs."),
        ("bronchus", "Each bronchus carries air into one lung."),
        ("bronchiole", "Bronchioles are small airways that lead to the alveoli."),
        ("alveoli", "Alveoli are tiny air sacs where oxygen and carbon dioxide are exchanged with the blood."),
        ("lung", "The lungs are the organs where gas exchange takes place."),
        ("diaphragm", "The diaphragm is a muscle that contracts to draw air into the lungs."),
        ("rib", "The ribs protect the lungs and move to change the volume of the chest."),
        ("nasal cavity", "The nasal cavity warms, moistens and filters the air."),
    ]),
    "mitochondrion": _t(["mitochondria", "mitochondrion structure"], "The mitochondrion is the organelle where most aerobic respiration takes place.", [
        ("outer membrane", "The outer membrane surrounds the mitochondrion and is smooth."),
        ("inner membrane", "The inner membrane is folded and holds the enzymes of the electron transport chain."),
        ("cristae", "Cristae are the folds of the inner membrane that give a large surface area."),
        ("matrix", "The matrix is the fluid inside, where the Krebs cycle takes place."),
        ("intermembrane space", "The intermembrane space is where hydrogen ions build up to drive ATP synthesis."),
        ("ribosome", "Mitochondrial ribosomes make some of the organelle's own proteins."),
    ]),
    "chloroplast": _t(["chloroplasts", "chloroplast structure"], "The chloroplast is the organelle where photosynthesis takes place.", [
        ("outer membrane", "The outer membrane surrounds the chloroplast."),
        ("inner membrane", "The inner membrane encloses the stroma."),
        ("stroma", "The stroma is the fluid where the Calvin cycle makes sugars."),
        ("thylakoid", "Thylakoids are flattened sacs whose membranes hold chlorophyll and carry out the light reactions."),
        ("granum", "A granum is a stack of thylakoids."),
        ("starch grain", "Starch grains store the sugars made by photosynthesis."),
    ]),
    "water cycle": _t(["the water cycle", "hydrological cycle"], "The water cycle describes how water moves between the surface, the air and the ground.", [
        ("evaporation", "Evaporation is when the sun's heat turns liquid water into water vapour."),
        ("condensation", "Condensation is when water vapour cools and turns into tiny droplets."),
        ("cloud", "Clouds are masses of tiny water droplets or ice crystals."),
        ("precipitation", "Precipitation is water falling as rain, hail or snow."),
        ("runoff", "Runoff is water flowing over the land into rivers and lakes."),
        ("infiltration", "Infiltration is water soaking into the soil and becoming groundwater."),
        ("transpiration", "Transpiration is water vapour lost from the leaves of plants."),
    ]),
    "fractional distillation": _t(["fractional distillation of crude oil", "crude oil distillation", "fractionating column"], "Fractional distillation separates crude oil into fractions with different boiling points.", [
        ("furnace", "The furnace heats the crude oil until most of it vaporises."),
        ("fractionating column", "The fractionating column is hotter at the bottom and cooler at the top."),
        ("refinery gas", "Refinery gases have the lowest boiling points and leave at the top."),
        ("petrol", "Petrol is a light fraction used as fuel for cars."),
        ("kerosene", "Kerosene is used as jet fuel."),
        ("diesel", "Diesel is used as fuel for lorries and some cars."),
        ("fuel oil", "Fuel oil is used in ships and power stations."),
        ("bitumen", "Bitumen has the highest boiling point and is used to make roads."),
    ], "chemistry"),
    "simple distillation": _t(["distillation", "distillation apparatus", "distillation set up", "laboratory distillation"], "Simple distillation separates a liquid from a solution by evaporating it and condensing the vapour.", [
        ("round-bottomed flask", "The round-bottomed flask holds the mixture that is heated."),
        ("thermometer", "The thermometer measures the temperature of the vapour."),
        ("condenser", "The condenser cools the vapour so that it turns back into a liquid."),
        ("water inlet", "Cold water enters the condenser at the water inlet at the bottom."),
        ("water outlet", "Warm water leaves the condenser at the water outlet at the top."),
        ("receiving flask", "The receiving flask collects the distillate."),
        ("heat source", "The heat source warms the mixture until it boils."),
    ], "chemistry"),
    "electrolysis": _t(["electrolysis apparatus", "electrolysis of brine", "electrolytic cell", "electrolysis set up"], "Electrolysis uses electricity to break down an ionic compound.", [
        ("power supply", "The power supply provides the direct current that drives electrolysis."),
        ("anode", "The anode is the positive electrode, where negative ions lose electrons."),
        ("cathode", "The cathode is the negative electrode, where positive ions gain electrons."),
        ("electrolyte", "The electrolyte is the molten or dissolved ionic compound that conducts electricity."),
        ("beaker", "The beaker holds the electrolyte."),
        ("connecting wires", "The connecting wires carry electrons around the circuit."),
    ], "chemistry"),
    "titration": _t(["titration apparatus", "acid base titration", "titration set up"], "A titration finds the volume of one solution that exactly reacts with another.", [
        ("burette", "The burette delivers the solution slowly and measures its volume."),
        ("conical flask", "The conical flask holds the solution being titrated, so it can be swirled."),
        ("indicator", "The indicator changes colour at the end point."),
        ("white tile", "The white tile makes the colour change easier to see."),
        ("burette stand", "The stand and clamp hold the burette upright."),
        ("tap", "The tap controls the flow of solution from the burette."),
    ], "chemistry"),
    "filtration": _t(["filtration apparatus", "filtering", "filtration set up"], "Filtration separates an insoluble solid from a liquid.", [
        ("funnel", "The funnel holds the filter paper."),
        ("filter paper", "The filter paper traps the solid and lets the liquid pass through."),
        ("residue", "The residue is the solid left on the filter paper."),
        ("filtrate", "The filtrate is the clear liquid that passes through."),
        ("conical flask", "The conical flask collects the filtrate."),
        ("retort stand", "The retort stand holds the funnel in place."),
    ], "chemistry"),
    "paper chromatography": _t(["chromatography", "chromatography apparatus"], "Paper chromatography separates the coloured substances in a mixture.", [
        ("chromatography paper", "The chromatography paper carries the solvent upwards by capillary action."),
        ("solvent", "The solvent moves up the paper and carries the substances with it."),
        ("baseline", "The baseline is the pencil line where the sample spots are placed."),
        ("sample spot", "Each sample spot is a small drop of the mixture to be separated."),
        ("solvent front", "The solvent front is the highest point reached by the solvent."),
        ("beaker", "The beaker holds the solvent and the paper."),
    ], "chemistry"),
    "bunsen burner": _t(["bunsen", "the bunsen burner"], "A Bunsen burner is a laboratory gas burner used for heating.", [
        ("chimney", "The chimney is the tube where the gas and air mix."),
        ("air hole", "The air hole lets air into the burner."),
        ("collar", "The collar is turned to open or close the air hole."),
        ("gas inlet", "The gas inlet connects the burner to the gas supply."),
        ("base", "The base keeps the burner stable."),
        ("flame", "The flame is hottest when the air hole is open."),
    ], "chemistry"),
    "blast furnace": _t(["blast furnace iron extraction", "extraction of iron"], "The blast furnace extracts iron from iron ore.", [
        ("raw materials", "Iron ore, coke and limestone are fed in at the top."),
        ("hot air blast", "Hot air is blown in near the bottom so that the coke burns."),
        ("waste gases", "Waste gases leave at the top."),
        ("molten slag", "Molten slag floats on top of the iron and is tapped off."),
        ("molten iron", "Molten iron collects at the bottom and is tapped off."),
    ], "chemistry"),
    "electrochemical cell": _t(["daniell cell", "simple cell", "galvanic cell", "voltaic cell"], "An electrochemical cell produces electricity from a chemical reaction.", [
        ("zinc electrode", "The zinc electrode is the negative terminal, where zinc loses electrons."),
        ("copper electrode", "The copper electrode is the positive terminal, where copper ions gain electrons."),
        ("zinc sulfate solution", "Zinc sulfate solution surrounds the zinc electrode."),
        ("copper sulfate solution", "Copper sulfate solution surrounds the copper electrode."),
        ("salt bridge", "The salt bridge lets ions move between the two solutions and completes the circuit."),
        ("voltmeter", "The voltmeter measures the voltage of the cell."),
        ("wire", "The wire carries electrons from the zinc to the copper."),
    ], "chemistry"),
    "atom": _t(["atomic structure", "bohr model", "structure of the atom", "structure of an atom"], "An atom has a small dense nucleus surrounded by electrons.", [
        ("nucleus", "The nucleus is at the centre of the atom and contains the protons and neutrons."),
        ("proton", "Protons are positively charged particles in the nucleus."),
        ("neutron", "Neutrons have no charge and are found in the nucleus."),
        ("electron", "Electrons are negatively charged and move around the nucleus."),
        ("electron shell", "Electrons are arranged in energy levels called shells."),
    ], "chemistry"),
}

LOOK = {  # one-line appearance hints for the tutor when it must pick a numbered region
    "nucleus": "a large round structure, often darker", "chloroplast": "an oval green structure",
    "vacuole": "a large pale sac", "mitochondrion": "a small oval with a folded inner membrane",
    "cell wall": "the thick outer layer around the cell", "cell membrane": "a thin line just inside the cell wall",
    "cytoplasm": "the jelly-like area that fills the cell", "ribosome": "a tiny dot", "aorta": "the largest tube leaving the heart",
    "septum": "the wall between the left and right sides", "lens": "a clear oval disc behind the pupil",
    "retina": "the layer lining the back of the eye", "burette": "a long thin graduated tube with a tap",
    "condenser": "a long tube surrounded by a jacket", "thermometer": "a thin glass tube with a bulb",
    "anode": "the positive electrode", "cathode": "the negative electrode",
    # digestive system
    "mouth": "the opening at the top, with teeth and a tongue", "oesophagus": "a long thin tube running from the throat down to the stomach",
    "stomach": "a J-shaped muscular bag below the oesophagus", "liver": "a large reddish-brown organ lying beside the stomach",
    "gall bladder": "a small green pear-shaped sac tucked under the liver", "pancreas": "a pale yellow elongated gland behind and below the stomach",
    "small intestine": "the long, tightly coiled tube in the middle of the abdomen", "large intestine": "the wider tube that frames the coiled small intestine",
    "rectum": "the short final section of the gut near the bottom", "anus": "the opening at the very bottom of the gut",
    # respiratory system
    "trachea": "a ribbed tube running down from the throat that splits in two", "lung": "a large spongy organ on each side of the chest",
    "bronchus": "one of the two tubes that carry air from the trachea into a lung", "alveoli": "tiny air sacs at the ends of the smallest airways",
    "diaphragm": "a dome-shaped sheet of muscle under the lungs",
    # plant cell / flower / eye
    "starch grain": "a small pale granule inside the cell", "petal": "a coloured leaf-like part of the flower",
    "stamen": "a thin stalk with a pollen-bearing tip", "stigma": "the sticky top of the central female part",
    "ovary": "the swollen base of the central female part", "cornea": "the clear dome at the very front of the eye",
    "iris": "the coloured ring around the pupil", "pupil": "the dark round opening in the middle of the iris",
    "optic nerve": "a thick cord leaving the back of the eye",
}


def curated_topic(topic: str) -> Optional[dict]:
    key = C._norm(topic)
    key = re.sub(r"^(the|a|an|human) ", "", key)
    for name, d in CURATED.items():
        names = {C._norm(name)} | {C._norm(a) for a in d["aliases"]}
        names |= {re.sub(r"^(the|a|an|human) ", "", n) for n in names}
        if key in names:
            return {"name": name, **d}
    for name, d in CURATED.items():                                          # looser: "cross section of a plant cell"
        names = [C._norm(name)] + [C._norm(a) for a in d["aliases"]]
        if any(n and (f" {n} " in f" {key} ") for n in names):
            return {"name": name, **d}
    return None


def all_part_names() -> set:
    """Every structure name in the built-in notes (used to notice a step that wanders into another organ)."""
    return {C._norm(p) for d in CURATED.values() for p, _ in d["parts"]}


def curated_source(topic: str, level_key: str) -> Optional[dict]:
    d = curated_topic(topic)
    if not d:
        return None
    n = C.get_level(level_key)["max_parts"]
    parts = d["parts"][:n]
    text = d["intro"] + " " + " ".join(s for _, s in d["parts"])
    return {"title": d["name"], "text": text, "url": None, "license": "Built-in notes", "source": "Built-in notes",
            "parts": [p for p, _ in parts], "sentences": {p: s for p, s in d["parts"]}, "curated": True, "subject": d["subject"]}


# =====================================================================================================
# Wikipedia (online, cached)
# =====================================================================================================
WIKI_API = "https://en.wikipedia.org/w/api.php"


def _wget(params: dict, timeout: int = 12) -> dict:
    r = requests.get(WIKI_API, params={**params, "format": "json", "formatversion": 2}, headers={"User-Agent": UA}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def wikipedia(topic: str, timeout: int = 12) -> Optional[dict]:
    data = _wget({"action": "query", "list": "search", "srsearch": topic, "srlimit": 1}, timeout)
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None
    title = hits[0]["title"]
    d = _wget({"action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "plain", "titles": title, "redirects": 1}, timeout)
    pages = d.get("query", {}).get("pages", [])
    text = (pages[0].get("extract") if pages else "") or ""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()[:9000]
    if len(text) < 200:
        return None
    return {"title": title, "text": text, "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
            "license": "CC BY-SA 4.0", "source": "Wikipedia"}


def fetch_topic(topic: str, level_key: str, store: Optional[SourceStore] = None, online: bool = True) -> Optional[dict]:
    """Built-in notes first, then cached/online Wikipedia.  Returns a source dict or None."""
    c = curated_source(topic, level_key)
    if c:
        return c
    key = "wiki:" + C._norm(topic)
    if store:
        hit = store.get(key)
        if hit:
            return {**hit, "from_cache": True}
    if not online:
        return None
    try:
        res = wikipedia(topic)
    except Exception:                                                        # noqa: BLE001  (offline, rate limit ...)
        return None
    if res and store:
        store.put(key, res)
    return res


# =====================================================================================================
# Reference pictures (Wikimedia Commons, cached).
#   The CODE fetches; the tutor never browses.  Only Wikimedia hosts are ever contacted, only freely licensed files are
#   used (no NonCommercial / NoDerivatives), and the licence + author travel with the picture for the on-screen credit.
#   The tutor's only job is to say how sure it is that the picture really shows the topic.
# =====================================================================================================
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
ALLOWED_HOSTS = {"commons.wikimedia.org", "upload.wikimedia.org"}
MAX_REF_BYTES = 3_000_000


def free_license(short: str) -> bool:
    s = (short or "").strip().lower()
    if not s or re.search(r"\b(nc|nd)\b|non-?commercial|no-?deriv|fair use|all rights", s):
        return False
    return s.startswith(("cc by", "cc-by", "cc0", "public domain", "pd"))


def _host_ok(url: str) -> bool:
    from urllib.parse import urlparse
    u = urlparse(url or "")
    return u.scheme == "https" and (u.hostname or "") in ALLOWED_HOSTS


def _strip_html(s: str) -> str:
    import html
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def rank_commons(pages: list[dict], topic: str) -> list[dict]:
    """Free, big-enough, drawn-or-photographed files, best title match first."""
    want = set(C.content_words(topic))
    scored = []
    for pg in pages:
        info = (pg.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        lic = (meta.get("LicenseShortName") or {}).get("value", "")
        if not (info.get("thumburl") and _host_ok(info["thumburl"]) and free_license(lic)):
            continue
        if info.get("mime") not in ("image/png", "image/jpeg", "image/svg+xml") or (info.get("width") or 0) < 500:
            continue
        title = pg.get("title", "")
        tw = set(C.content_words(re.sub(r"^File:|\.\w+$", "", title)))
        score = len(want & tw) * 2 + (2 if re.search(r"diagram|labell?ed|anatomy|illustration|structure", title, re.I) else 0)
        scored.append((score, pg, info, lic, (meta.get("Artist") or {}).get("value", "")))
    scored.sort(key=lambda t: -t[0])
    return [{"title": pg["title"], "page_url": info.get("descriptionurl", ""), "thumb_url": info["thumburl"], "license": lic,
             "author": _strip_html(artist)[:80] or "unknown author"} for sc, pg, info, lic, artist in scored if sc > 0]


def _download_thumb(url: str, timeout: int = 15) -> Optional[str]:
    """Download one allowed picture and return it as a base64 JPEG (max 800 px), or None."""
    import base64
    import io
    from PIL import Image
    if not _host_ok(url):
        return None
    r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout, allow_redirects=False)
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/") or len(r.content) > MAX_REF_BYTES:
        return None
    im = Image.open(io.BytesIO(r.content)).convert("RGB")
    im.thumbnail((800, 800))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def commons_references(topic: str, limit: int = 3, timeout: int = 12) -> list[dict]:
    r = requests.get(COMMONS_API, params={
        "action": "query", "format": "json", "formatversion": 2, "generator": "search", "gsrnamespace": 6, "gsrlimit": 12,
        "gsrsearch": f"{topic} diagram filetype:bitmap|drawing", "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 800, "iiextmetadatafilter": "LicenseShortName|Artist"}, headers={"User-Agent": UA}, timeout=timeout)
    r.raise_for_status()
    out = []
    for cand in rank_commons(r.json().get("query", {}).get("pages", []), topic)[:limit + 2]:
        try:
            b64 = _download_thumb(cand["thumb_url"], timeout)
        except Exception:                                                        # noqa: BLE001
            b64 = None
        if b64:
            out.append({**{k: v for k, v in cand.items() if k != "thumb_url"}, "b64": b64})
        if len(out) >= limit:
            break
    return out


def fetch_references(topic: str, store: Optional[SourceStore] = None, online: bool = True) -> list[dict]:
    """Reference pictures for a topic: cache first (works offline), then Commons.  [] when none / offline."""
    key = "commons:" + C._norm(topic)
    if store:
        hit = store.get(key)
        if hit is not None:
            return hit
    if not online:
        return []
    try:
        refs = commons_references(topic)
    except Exception:                                                            # noqa: BLE001  (offline, rate limit ...)
        return []
    if store:
        store.put(key, refs)
    return refs


def credit_line(ref: dict) -> str:
    return f"Reference picture: {ref.get('title', '')} by {ref.get('author', 'unknown author')} ({ref.get('license', '')}), Wikimedia Commons"


# =====================================================================================================
# PubChem (name -> SMILES), cached
# =====================================================================================================
def pubchem_smiles(name: str, store: Optional[SourceStore] = None, timeout: int = 10) -> Optional[str]:
    key = "pubchem:" + C._norm(name)
    if store:
        hit = store.get(key)
        if hit is not None:
            return hit or None
    try:
        r = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/property/CanonicalSMILES/JSON",
                         headers={"User-Agent": UA}, timeout=timeout)
        if r.status_code != 200:
            if r.status_code == 404 and store:
                store.put(key, "")
            return None
        props = r.json()["PropertyTable"]["Properties"][0]
        smi = next((v for k, v in props.items() if "SMILES" in k and isinstance(v, str)), None)
    except Exception:                                                        # noqa: BLE001
        return None
    if smi and store:
        store.put(key, smi)
    return smi


# =====================================================================================================
# Curated processes (for infographics).  step = (title, one sentence, scene for the picture, first level that shows it)
#   level 1 = Ordinary and above, 2 = Advanced and above, 3 = University only
# =====================================================================================================
def _p(aliases, shape, intro, steps, subject="", domain="biology", system=""):
    """system = the curated diagram topic the process happens INSIDE; steps are then drawn on that one diagram."""
    return {"name": aliases[0], "aliases": aliases, "shape": shape, "intro": intro, "steps": steps, "subject": subject, "domain": domain, "system": system}


CURATED_PROCESSES = [
    _p(["photosynthesis", "photosynthesis process"], "io",
       "Photosynthesis is how green plants make food using light energy.",
       [("Sunlight arrives", "Sunlight shines on the green leaves of the plant.", "sunlight shining on a green leaf", 1),
        ("Water is absorbed", "Roots take in water from the soil and it travels up to the leaves.", "plant roots absorbing water from soil, water moving up the stem", 1),
        ("Carbon dioxide enters", "Carbon dioxide from the air enters the leaf through tiny pores called stomata.", "carbon dioxide entering a leaf through small pores", 1),
        ("Chlorophyll traps light", "Chlorophyll in the chloroplasts absorbs light energy.", "green chloroplasts inside a leaf cell absorbing light", 2),
        ("Glucose is made", "The plant uses the light energy to turn water and carbon dioxide into glucose.", "glucose sugar molecule forming inside a leaf cell", 1),
        ("Oxygen is released", "Oxygen is released into the air as a by-product.", "oxygen bubbles leaving a green leaf", 1),
        ("Glucose is used or stored", "The glucose is used for energy or stored as starch.", "plant storing starch in a root and using energy to grow", 2)], "a green plant leaf"),
    _p(["water cycle", "hydrological cycle"], "cycle",
       "The water cycle is the continuous movement of water between the Earth's surface and the atmosphere.",
       [("Evaporation", "The sun heats water in oceans and rivers and it evaporates into water vapour.", "sun heating a lake with water vapour rising", 1),
        ("Transpiration", "Plants also lose water vapour to the air through their leaves.", "trees releasing water vapour into the air", 2),
        ("Condensation", "The vapour rises, cools and condenses into tiny droplets that form clouds.", "clouds forming high in the sky", 1),
        ("Precipitation", "Water falls from the clouds as rain, hail or snow.", "rain falling from dark clouds onto hills", 1),
        ("Run-off", "Rain flows over the land into streams and rivers.", "rain water flowing down a hill into a river", 1),
        ("Infiltration", "Some water soaks into the soil and becomes groundwater.", "water soaking down through soil layers to groundwater", 2),
        ("Collection", "Rivers and groundwater carry the water back to the sea or lakes, ready to evaporate again.", "river flowing into the sea", 1)]),
    _p(["carbon cycle"], "cycle",
       "The carbon cycle moves carbon between the air, living things, soil and oceans.",
       [("Photosynthesis", "Plants take carbon dioxide from the air and use it to make food.", "green plants taking in carbon dioxide from the air", 1),
        ("Feeding", "Animals get carbon by eating plants or other animals.", "a cow grazing on grass", 1),
        ("Respiration", "Plants and animals release carbon dioxide into the air when they respire.", "animal breathing out carbon dioxide into the air", 1),
        ("Decomposition", "Decomposers break down dead plants and animals and release carbon dioxide.", "mushrooms decomposing a fallen log", 1),
        ("Fossil fuel formation", "Over millions of years some dead matter becomes coal, oil and gas.", "layers of buried dead plants turning into coal and oil underground", 2),
        ("Combustion", "Burning wood and fossil fuels releases carbon dioxide into the air.", "smoke rising from a burning fire and a factory chimney", 1)]),
    _p(["nitrogen cycle"], "cycle",
       "The nitrogen cycle changes nitrogen between forms that living things can and cannot use.",
       [("Nitrogen fixation", "Bacteria and lightning change nitrogen gas from the air into nitrates and ammonium.", "lightning and root nodule bacteria fixing nitrogen", 1),
        ("Nitrification", "Nitrifying bacteria in the soil turn ammonium into nitrates.", "bacteria in soil turning ammonium into nitrates", 2),
        ("Absorption by plants", "Plants absorb nitrates from the soil and use them to make proteins.", "plant roots absorbing nitrates from soil", 1),
        ("Feeding", "Animals get nitrogen by eating plants or other animals.", "a goat eating leaves", 1),
        ("Decay", "Decomposers break down dead matter and waste and release ammonium.", "decomposers breaking down a dead animal on soil", 1),
        ("Denitrification", "Denitrifying bacteria turn nitrates back into nitrogen gas that returns to the air.", "bacteria in waterlogged soil releasing nitrogen gas into the air", 2)]),
    _p(["digestion", "digestion of food", "human digestion"], "linear",
       "Digestion breaks food into small soluble molecules that the body can absorb.",
       [("Ingestion", "Food is taken into the mouth where teeth chew it.", "a person chewing food, teeth visible", 1),
        ("Saliva", "Saliva mixes with food and starts to break down starch.", "mouth with saliva and a food ball", 2),
        ("Swallowing", "The food ball moves down the oesophagus by peristalsis.", "food moving down the oesophagus", 1),
        ("Stomach", "The stomach churns food with acid and enzymes that start to digest protein.", "stomach churning food with acid", 1),
        ("Small intestine", "Enzymes finish digestion and the small molecules are absorbed into the blood.", "small intestine absorbing nutrients into blood vessels", 1),
        ("Large intestine", "The large intestine absorbs water from the undigested food.", "large intestine absorbing water", 1),
        ("Egestion", "The remaining waste leaves the body through the anus.", "waste leaving the body, simple anatomical drawing", 1)], "", "biology", "digestive system"),
    _p(["mitosis", "cell division by mitosis"], "cycle",
       "Mitosis is cell division that makes two genetically identical daughter cells.",
       [("Interphase", "The cell grows and copies its DNA.", "a round cell with a nucleus and thread-like chromatin", 1),
        ("Prophase", "The chromosomes shorten and thicken and the nuclear membrane breaks down.", "a cell with visible thick chromosomes and no nuclear membrane", 1),
        ("Metaphase", "The chromosomes line up across the middle of the cell.", "a cell with chromosomes lined up in the middle", 1),
        ("Anaphase", "The chromatids are pulled apart to opposite ends of the cell.", "a cell with chromosomes pulled to opposite poles", 1),
        ("Telophase", "Two new nuclei form at the ends of the cell.", "a cell with two nuclei forming at opposite ends", 2),
        ("Cytokinesis", "The cytoplasm splits and two identical daughter cells are formed.", "one cell splitting into two identical cells", 1)]),
    _p(["haber process"], "io",
       "The Haber process makes ammonia from nitrogen and hydrogen.",
       [("Raw materials", "Nitrogen from the air and hydrogen from natural gas are the raw materials.", "two gas cylinders labelled by colour, one blue one white, next to a pipe", 1),
        ("Compression", "The gases are mixed and compressed to about 200 atmospheres.", "a compressor squeezing gas into a strong steel pipe", 2),
        ("Reactor with iron catalyst", "The gases pass over an iron catalyst at about 450 degrees Celsius.", "a tall steel reactor with a catalyst bed and a flame heating it", 1),
        ("Ammonia forms", "Some nitrogen and hydrogen react reversibly to form ammonia.", "gas molecules in a reactor combining into ammonia", 1),
        ("Cooling", "The mixture is cooled so the ammonia turns into a liquid.", "cooling tower condensing a gas into a liquid", 1),
        ("Recycling", "Unreacted nitrogen and hydrogen are recycled back into the reactor.", "pipe carrying leftover gases back to the reactor", 1)], "", "chemistry"),
    _p(["contact process"], "io",
       "The contact process makes sulfuric acid.",
       [("Burn sulfur", "Sulfur is burned in air to make sulfur dioxide.", "burning yellow sulfur in a furnace", 1),
        ("Purify the gas", "The sulfur dioxide is cleaned and dried.", "gas passing through a cleaning chamber", 2),
        ("Catalytic converter", "Sulfur dioxide and oxygen pass over a vanadium oxide catalyst at about 450 degrees Celsius to make sulfur trioxide.", "tall converter with several catalyst trays", 1),
        ("Absorption", "The sulfur trioxide is absorbed in concentrated sulfuric acid to make oleum.", "gas bubbling into a tank of acid", 1),
        ("Dilution", "The oleum is diluted with water to make concentrated sulfuric acid.", "adding water carefully to a tank of oleum", 1)], "", "chemistry"),
    _p(["blood flow through the heart", "blood flows through the heart", "how blood flows through the heart", "how the heart pumps blood", "circulation through the heart", "heart pumping"], "cycle",
       "Blood passes through the four chambers of the heart in a fixed order.",
       [("Blood enters the right side", "Deoxygenated blood from the body flows through the vena cava into the right atrium.", "blood entering the right atrium", 1),
        ("Through the tricuspid valve", "The right atrium contracts and pushes the blood through the tricuspid valve into the right ventricle.", "blood passing the tricuspid valve", 1),
        ("Pumped towards the lungs", "The right ventricle pumps the blood into the pulmonary artery, which leaves the heart for the lungs.", "blood leaving the right ventricle", 1),
        ("Blood returns from the lungs", "Oxygenated blood from the lungs flows through the pulmonary vein into the left atrium.", "blood entering the left atrium", 1),
        ("Through the bicuspid valve", "The left atrium contracts and pushes the blood through the bicuspid valve into the left ventricle.", "blood passing the bicuspid valve", 1),
        ("Pumped to the body", "The left ventricle contracts strongly and pumps the blood into the aorta, which carries it to the body.", "blood leaving the left ventricle", 1)],
       "", "biology", "human heart"),
    _p(["nerve impulse along a neuron", "how a neuron carries an impulse", "nerve impulse", "impulse transmission"], "linear",
       "A nerve impulse travels along one neuron in one direction.",
       [("Dendrites receive", "The dendrites receive a signal from another neuron and carry it to the cell body.", "dendrites receiving a signal", 1),
        ("Cell body", "The cell body combines the signals and passes the impulse on to the axon.", "impulse in the cell body", 1),
        ("Along the axon", "The impulse travels along the axon, which is a long fibre.", "impulse moving along the axon", 1),
        ("Jumping the gaps", "The myelin sheath insulates the axon, so the impulse jumps between the nodes of Ranvier and travels faster.", "impulse jumping between nodes of Ranvier", 2),
        ("Axon terminals", "At the axon terminals the impulse passes the signal on to the next cell.", "signal leaving the axon terminals", 1)],
       "", "biology", "neuron"),
    _p(["reflex action", "reflex arc process", "how a reflex works"], "linear",
       "A reflex is a fast automatic response that does not wait for the brain to decide.",
       [("Stimulus detected", "The receptor detects the stimulus and starts an impulse.", "receptor detecting a stimulus", 1),
        ("Sensory neuron", "The impulse travels along the sensory neuron to the spinal cord.", "impulse along the sensory neuron", 1),
        ("Relay neuron", "In the spinal cord the relay neuron passes the impulse to the motor neuron.", "impulse crossing the relay neuron", 1),
        ("Motor neuron", "The impulse travels along the motor neuron to the effector.", "impulse along the motor neuron", 1),
        ("Response", "The effector, a muscle, contracts and produces the response.", "effector producing a response", 1)],
       "", "biology", "reflex arc"),
    _p(["how the brain handles a stimulus", "brain response to a stimulus", "how the brain processes information", "information processing in the brain", "signal pathway in the brain"], "linear",
       "Information that reaches the brain is passed between its parts in a set order.",
       [("Signals arrive", "Nerve signals from the body enter the brain through the brain stem.", "signals entering the brain stem", 1),
        ("Relay", "The brain stem passes the signals up to the thalamus, which relays them.", "signals going from the brain stem to the thalamus", 1),
        ("Sent to the cerebrum", "The thalamus sends the signals to the correct part of the cerebrum.", "signals going from the thalamus to the cerebrum", 1),
        ("Understanding", "The cerebrum interprets the signals and decides what response is needed.", "cerebrum interpreting signals", 1),
        ("Coordination", "The cerebrum sends the plan to the cerebellum, which coordinates smooth and balanced movement.", "plan going from the cerebrum to the cerebellum", 2),
        ("Response leaves", "The finished instructions pass down through the brain stem and leave the brain.", "instructions going down the brain stem", 1)],
       "", "biology", "human brain"),
    _p(["seed germination", "germination"], "linear",
       "Germination is when a seed starts to grow into a young plant.",
       [("Dry seed", "A dry seed waits with a small embryo inside.", "a dry bean seed on soil", 1),
        ("Water is absorbed", "The seed absorbs water and swells.", "a bean seed swelling in moist soil", 1),
        ("Radicle emerges", "The radicle, the first root, grows down into the soil.", "a seed with a small root growing downward", 1),
        ("Plumule grows", "The plumule, the first shoot, grows upward towards the light.", "a seedling with a shoot growing up", 1),
        ("Leaves open", "The first leaves open and the seedling begins to photosynthesise.", "a small seedling with open green leaves in sunlight", 1)]),
    _p(["pollination and fertilisation", "pollination", "fertilisation in plants"], "linear",
       "Pollination and fertilisation lead to the formation of seeds in flowering plants.",
       [("Pollen is produced", "Pollen grains are made in the anthers of the flower.", "flower with yellow pollen on the anthers", 1),
        ("Pollination", "Pollen is carried by insects or wind to the stigma of a flower.", "a bee carrying pollen to a flower", 1),
        ("Pollen tube grows", "A pollen tube grows down the style towards the ovule.", "pollen tube growing down the style of a flower", 2),
        ("Fertilisation", "The male nucleus joins with the female nucleus in the ovule.", "male and female cells joining inside an ovule", 1),
        ("Seed and fruit form", "The ovule becomes a seed and the ovary becomes a fruit.", "a flower turning into a fruit with seeds", 1)]),
]

_PROC_BY_NAME = {n.lower(): p for p in CURATED_PROCESSES for n in p["aliases"]}


def curated_process(topic: str, level_key: str) -> Optional[dict]:
    key = C._norm(re.sub(r"^(the |a |an )", "", (topic or "").lower().strip()))
    hit = None
    for name, p in _PROC_BY_NAME.items():
        n = C._norm(name)
        if key == n or key.endswith(" " + n) or n in key.split(" of ")[-1:][0:1] or (len(n) > 5 and n in key):
            hit = p
            break
    if not hit:
        return None
    lvl = {"ordinary": 1, "advanced": 2, "university": 3}.get(level_key, 1)
    steps = [{"title": t, "text": s, "scene": sc} for (t, s, sc, m) in hit["steps"] if m <= lvl]
    text = hit["intro"] + " " + " ".join(s["text"] for s in steps)
    return {"title": hit["name"].title(), "shape": hit["shape"], "steps": steps, "text": text, "subject": hit.get("subject", ""),
            "source": "Built-in notes", "license": "Built-in notes", "url": None, "curated": True, "domain": hit["domain"], "system": hit.get("system", "")}
