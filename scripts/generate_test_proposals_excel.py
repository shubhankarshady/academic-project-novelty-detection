import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.api.service import get_novelty_service

service = get_novelty_service()

test_cases = [
    {
        "test_id": "TEST-01",
        "expected_novelty_tier": "High Novelty (Original Work)",
        "expected_score_range": "90% - 98%",
        "domain": "Quantum Computing & Cryptography",
        "rationale": "Cutting-edge topic; zero prior overlap in corpus with photonic quantum circuits.",
        "project_title": "Quantum Neural Network for Photonic Cryptographic Key Distribution Under High Noise",
        "abstract": "We propose a hybrid quantum-classical neural network architecture designed to reconstruct entangled photon polarization states in continuous-variable quantum key distribution (CV-QKD) systems subjected to atmospheric turbulence.",
        "problem_statement": "Atmospheric turbulence and decoherence cause significant phase drift and photon loss in free-space quantum key distribution, degrading secret key rates below acceptable cryptographic thresholds.",
        "objectives": "Design a parameterized quantum circuit (PQC) coupled with a classical variational layer to compensate for real-time channel noise without requiring active laser calibration hardware.",
        "proposed_methodology": "Simulating multi-mode Gaussian states in photonic circuits, optimizing quantum gate angles via Adam gradient descent on the Pennylane framework, and testing across simulated free-space channels.",
        "dataset": "Photonic Entanglement Experimental Lab Benchmarks",
        "algorithm_model": "Parameterized Quantum Circuit (PQC) with Variational Quantum Eigensolver (VQE)",
        "technology_tools": "Pennylane, Qiskit, PyTorch, Python 3.10"
    },
    {
        "test_id": "TEST-02",
        "expected_novelty_tier": "High Novelty (Original Work)",
        "expected_score_range": "85% - 95%",
        "domain": "Bioacoustics & Radar Sensing",
        "rationale": "Novel dual-sensor fusion (mmWave radar + audio) for nocturnal ecology; no similar project exists in reference corpus.",
        "project_title": "Multimodal Micro-Doppler Radar and Acoustic Fusion for Nocturnal Avian Bioacoustics",
        "abstract": "Monitoring nocturnal bird migrations requires resilient sensing unaffected by darkness or adverse weather. We introduce a dual-stream cross-attention transformer combining 24GHz FMCW radar micro-Doppler spectrograms with ultrasonic bioacoustic recordings.",
        "problem_statement": "Visual camera traps fail during night migration, and single-channel audio recording suffers from severe acoustic attenuation and ambient wind distortion.",
        "objectives": "Fuse RF radar micro-Doppler wing-beat signatures with directional microphone array audio to classify migratory bird species with high temporal resolution.",
        "proposed_methodology": "Dual-stream transformer backbone with cross-modal attention layers, temporal synchronization of radar chirps and audio WAV streams, evaluated on outdoor flight enclosures.",
        "dataset": "24GHz FMCW Radar Micro-Doppler and Audio Field Recordings",
        "algorithm_model": "Dual-Stream Cross-Attention Transformer (CrossViT)",
        "technology_tools": "PyTorch, Librosa, mmWave Studio, Python, CUDA"
    },
    {
        "test_id": "TEST-03",
        "expected_novelty_tier": "Moderate Novelty",
        "expected_score_range": "65% - 80%",
        "domain": "Bioinformatics & Graph ML",
        "rationale": "Applies Graph Neural Networks to clinical pharmacology; moderate novelty due to general ML healthcare overlap but distinct chemical graph formulation.",
        "project_title": "Explainable Graph Neural Networks for Pharmacovigilance and Adverse Drug-Drug Interaction Discovery",
        "abstract": "Adverse drug reactions resulting from polypharmacy represent a major clinical hazard. We develop an explainable multi-relational graph convolutional network that models molecular graph structures alongside biological protein target networks.",
        "problem_statement": "Clinical drug interaction prediction methods fail to provide molecular-level biological explanations for predicted adverse events, restricting physician trust in AI recommendations.",
        "objectives": "Predict pairwise drug-drug adverse interactions while extracting high-attention subgraphs indicating chemical pharmacophores responsible for toxicity.",
        "proposed_methodology": "Construct heterogeneous biomedical knowledge graphs from DrugBank and TWOSIDES, apply Relational GCN with integrated gradients for subgraph attribution.",
        "dataset": "DrugBank molecular repository and TWOSIDES adverse interaction database",
        "algorithm_model": "Relational Graph Convolutional Network (R-GCN) with Integrated Gradients",
        "technology_tools": "PyTorch Geometric, RDKit, NetworkX, Scikit-learn"
    },
    {
        "test_id": "TEST-04",
        "expected_novelty_tier": "Moderate to Low Novelty",
        "expected_score_range": "45% - 60%",
        "domain": "Cybersecurity & Web Safety",
        "rationale": "Uses Kaggle phishing dataset which is in the corpus (SYN-0003); moderate/low novelty because problem and dataset heavily overlap.",
        "project_title": "Automated Phishing Website Detection Using Hybrid Random Forest and URL N-grams",
        "abstract": "Phishing attacks remain a primary threat in web browsing security. This work implements an automated detection pipeline that extracts lexical, host-based, and n-gram features from suspicious website URLs and classifies them.",
        "problem_statement": "Manual verification of malicious URLs is slow, and blacklists fail against newly registered zero-day phishing landing pages.",
        "objectives": "Extract 30 lexical and structural URL attributes to distinguish legitimate domains from fraudulent phishing sites in real-time.",
        "proposed_methodology": "Lexical tokenization of URL strings, extraction of domain age and SSL certificates, followed by training Random Forest and XGBoost classifiers.",
        "dataset": "Kaggle phishing dataset",
        "algorithm_model": "Random Forest Classifier and XGBoost",
        "technology_tools": "Scikit-learn, Pandas, Beautiful Soup, Python"
    },
    {
        "test_id": "TEST-05",
        "expected_novelty_tier": "Low Novelty (Significant Prior Work)",
        "expected_score_range": "30% - 45%",
        "domain": "Agricultural Computer Vision",
        "rationale": "Near duplicate of SYN-0001; exact same PlantVillage dataset, same leaf disease problem, and standard CNN algorithm.",
        "project_title": "Plant Disease Identification and Leaf Blight Classification Using Convolutional Neural Networks",
        "abstract": "Crop diseases threaten global agricultural yields. This project applies deep Convolutional Neural Networks to classify tomato, potato, and apple leaf diseases from digital photographs of plant foliage.",
        "problem_statement": "Farmers lack expert agronomist access to diagnose fungal and bacterial plant leaf infections in early cultivation stages.",
        "objectives": "Automate multi-class classification of healthy versus infected plant leaves with high accuracy to assist farmers via a mobile application.",
        "proposed_methodology": "Image preprocessing, data augmentation (rotation, zooming, flipping), and training standard CNN architectures on leaf image datasets.",
        "dataset": "PlantVillage",
        "algorithm_model": "Convolutional Neural Network (CNN)",
        "technology_tools": "TensorFlow, Keras, OpenCV, NumPy, Matplotlib"
    },
    {
        "test_id": "TEST-06",
        "expected_novelty_tier": "Low Novelty (Significant Prior Work)",
        "expected_score_range": "30% - 45%",
        "domain": "Network Security & Intrusion Detection",
        "rationale": "Direct overlap with SYN-0002; uses CIC-IDS2017 dataset with standard Decision Tree / Random Forest for traffic classification.",
        "project_title": "Network Intrusion Detection System Using Machine Learning Techniques",
        "abstract": "Cyber attacks on corporate networks are escalating in volume. We evaluate standard machine learning algorithms including Decision Trees and Random Forests to detect DDoS and port scanning attacks in network packet flows.",
        "problem_statement": "Traditional rule-based intrusion detection systems fail to detect polymorphic malware and novel packet flooding variations.",
        "objectives": "Clean and normalize network packet flow features, apply feature selection, and evaluate binary attack detection accuracy.",
        "proposed_methodology": "Min-max normalization of PCAP packet statistics, recursive feature elimination, and classification with Decision Trees and Random Forest.",
        "dataset": "CIC-IDS2017",
        "algorithm_model": "Decision Tree and Random Forest Classifier",
        "technology_tools": "Python, Scikit-learn, Wireshark, Pandas"
    }
]

results = []
for tc in test_cases:
    eval_res = service.score_proposal(tc, top_k=1)
    top_match = eval_res["top_matches"][0] if eval_res.get("top_matches") else {}
    results.append({
        "Test ID": tc["test_id"],
        "Domain": tc["domain"],
        "Project Title": tc["project_title"],
        "Expected Novelty Tier": tc["expected_novelty_tier"],
        "Expected Score Range": tc["expected_score_range"],
        "Actual Predicted Novelty %": round(eval_res["overall_novelty_score_pct"], 2),
        "Actual Predicted Tier": eval_res["novelty_assessment_tier"],
        "Predicted Similarity (0-4)": round(eval_res["max_similarity_score"], 3),
        "Top Matched Corpus Project": f"{top_match.get('historical_project_id')}: {top_match.get('historical_project_title')}",
        "Key Matching Sections": ", ".join([f"{k} ({v:.2f})" for k, v in top_match.get("section_similarity_scores", {}).items() if v >= 0.20]),
        "Why Model Scored This": tc["rationale"],
        "Dataset": tc["dataset"],
        "Algorithm": tc["algorithm_model"],
        "Abstract": tc["abstract"],
        "Problem Statement": tc["problem_statement"],
        "Objectives": tc["objectives"],
        "Proposed Methodology": tc["proposed_methodology"],
        "Technology Tools": tc["technology_tools"]
    })

df_res = pd.DataFrame(results)
excel_path = ROOT / "data" / "new_test_proposals_with_expected_outputs.xlsx"
df_res.to_excel(excel_path, index=False, sheet_name="Test_Proposals_Evaluation")
print(f"Successfully generated Excel file at: {excel_path}")

for r in results:
    print(f"[{r['Test ID']}] {r['Project Title'][:45]}... -> {r['Actual Predicted Tier']} ({r['Actual Predicted Novelty %']}%)")
