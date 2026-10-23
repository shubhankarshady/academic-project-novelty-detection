/**
 * Formatting and helper utilities.
 */

export const SECTION_CONFIG = [
  {
    key: 'project_title',
    label: 'Project Title',
    minLength: 3,
    rows: 1,
    placeholder: 'e.g. Federated Learning for Privacy-Preserving Medical Diagnostics',
    hint: 'A descriptive title summarizing the academic research initiative.',
    required: true,
  },
  {
    key: 'domain',
    label: 'Domain / Research Area',
    minLength: 0,
    rows: 1,
    placeholder: 'e.g. Healthcare AI, Natural Language Processing, Robotics',
    hint: 'Optional domain or department classification (defaults to General).',
    required: false,
  },
  {
    key: 'abstract',
    label: 'Abstract',
    minLength: 10,
    rows: 3,
    placeholder: 'Comprehensive high-level summary of the academic proposal...',
    hint: 'Summarize the core research scope, approach, and primary expected contribution.',
    required: true,
  },
  {
    key: 'problem_statement',
    label: 'Problem Statement',
    minLength: 10,
    rows: 3,
    placeholder: 'Clearly articulate the specific technical problem and current limitations...',
    hint: 'Detail the domain challenges, gaps in existing literature, or practical limitations.',
    required: true,
  },
  {
    key: 'objectives',
    label: 'Objectives',
    minLength: 5,
    rows: 3,
    placeholder: 'Itemize the measurable research aims and goals...',
    hint: 'List specific technical aims, performance targets, and deliverables.',
    required: true,
  },
  {
    key: 'proposed_methodology',
    label: 'Proposed Methodology',
    minLength: 10,
    rows: 4,
    placeholder: 'Describe the technical workflow, data pipelines, model pipelines, and evaluation...',
    hint: 'Detailed architecture, algorithmic formulations, and implementation pipeline.',
    required: true,
  },
  {
    key: 'dataset',
    label: 'Dataset & Benchmarks',
    minLength: 3,
    rows: 2,
    placeholder: 'e.g. BraTS multi-institutional magnetic resonance imaging benchmarks, ImageNet, PhysioNet',
    hint: 'Names of open or private datasets, benchmarks, collection strategies, or synthetic generators.',
    required: true,
  },
  {
    key: 'algorithm_model',
    label: 'Algorithms & Models',
    minLength: 3,
    rows: 2,
    placeholder: 'e.g. 3D U-Net, ResNet-50, FedAvg with Differential Privacy noise',
    hint: 'Specific machine learning architectures, statistical models, or optimization algorithms.',
    required: true,
  },
  {
    key: 'technology_tools',
    label: 'Technologies & Tools',
    minLength: 2,
    rows: 2,
    placeholder: 'e.g. PyTorch, Flower Federated Framework, Python 3.10, CUDA',
    hint: 'Software libraries, hardware accelerators, simulation environments, and frameworks.',
    required: true,
  },
];

/**
 * Clean human-readable label for a section key.
 */
export function getSectionLabel(key) {
  const cfg = SECTION_CONFIG.find(c => c.key === key);
  if (cfg) return cfg.label;
  return key
    .split('_')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

/**
 * Format score percentage safely.
 */
export function formatPercent(value) {
  if (typeof value !== 'number' || isNaN(value)) return '0.0%';
  return `${value.toFixed(1)}%`;
}

/**
 * Format similarity score (0.00 to 1.00).
 */
export function formatScore(value) {
  if (typeof value !== 'number' || isNaN(value)) return '0.000';
  return value.toFixed(3);
}

/**
 * Format ISO timestamp into clean local date/time string.
 */
export function formatDate(isoString) {
  if (!isoString) return 'N/A';
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return isoString;
  }
}

/**
 * Assessment tier badge configuration.
 */
export function getTierBadgeProps(tier) {
  const normalized = (tier || '').toLowerCase();
  if (normalized.includes('high')) {
    return {
      label: tier || 'High Estimated Novelty',
      bgClass: 'bg-emerald-50 text-emerald-800 border-emerald-200',
      dotClass: 'bg-emerald-500',
      progressClass: 'bg-emerald-500',
    };
  }
  if (normalized.includes('moderate') || normalized.includes('medium')) {
    return {
      label: tier || 'Moderate Estimated Novelty',
      bgClass: 'bg-amber-50 text-amber-800 border-amber-200',
      dotClass: 'bg-amber-500',
      progressClass: 'bg-amber-500',
    };
  }
  return {
    label: tier || 'Low Estimated Novelty',
    bgClass: 'bg-rose-50 text-rose-800 border-rose-200',
    dotClass: 'bg-rose-500',
    progressClass: 'bg-rose-500',
  };
}

/**
 * Sample proposal for quick demonstration.
 */
export const SAMPLE_PROPOSALS = [
  {
    name: 'Healthcare AI: Federated MRI Classification (Recommended)',
    data: {
      project_title: 'Federated Learning for Privacy-Preserving Medical Diagnostics',
      domain: 'Healthcare AI',
      abstract: 'We propose a federated learning framework using convolutional neural networks to train diagnostic models across distributed hospital datasets without centralizing sensitive patient records.',
      problem_statement: 'Medical imaging datasets are siloed due to strict HIPAA regulations and patient confidentiality concerns, restricting large-scale multi-institutional AI model training.',
      objectives: 'Train high-accuracy classification models across multi-site hospitals while maintaining patient privacy and validating differential privacy bounds.',
      proposed_methodology: 'Implement FedAvg aggregation with differential privacy noise simulated across multiple client nodes and robust weight update validation.',
      dataset: 'BraTS multi-institutional magnetic resonance imaging benchmarks.',
      algorithm_model: '3D U-Net and ResNet-50 Convolutional Neural Networks.',
      technology_tools: 'PyTorch, Flower Federated Framework, Python, CUDA.',
    },
  },
  {
    name: 'Edge IoT: TinyML Vibration Anomaly Detection',
    data: {
      project_title: 'TinyML Anomaly Detection for Industrial Rotational Machinery via Acoustic Emissions',
      domain: 'Edge Computing & IoT',
      abstract: 'An ultra-low-power TinyML architecture deployed on ARM Cortex-M4 microcontrollers for real-time anomaly detection in high-speed industrial bearings using quantized autoencoders.',
      problem_statement: 'Industrial gearboxes experience catastrophic failures without continuous monitoring, yet cloud telemetry incurs latency and high cellular bandwidth overhead.',
      objectives: 'Design an 8-bit quantized autoencoder under 64KB RAM capable of sub-10ms vibration anomaly inference on microcontrollers.',
      proposed_methodology: 'Capture raw high-frequency accelerometer streams, compute Mel-frequency energy features on-chip, and pass through an INT8 compressed bottleneck autoencoder.',
      dataset: 'Case Western Reserve University (CWRU) Bearing Data Center benchmarks.',
      algorithm_model: 'Quantized Bottleneck Deep Autoencoder with dynamic reconstruction error thresholding.',
      technology_tools: 'TensorFlow Lite for Microcontrollers, C++, STM32CubeIDE, Python.',
    },
  },
];
