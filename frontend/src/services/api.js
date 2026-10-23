/**
 * Centralized API client for Academic Project Novelty Detection.
 * Integrates with FastAPI backend endpoints:
 * - GET  /api/health
 * - POST /api/score
 * - POST /api/save
 * - GET  /api/proposals
 * - GET  /api/proposals/{project_id}
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

class ApiError extends Error {
  constructor(message, status, details = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

/**
 * Helper to handle fetch responses and standardize error messages.
 */
async function handleResponse(response) {
  if (!response.ok) {
    let errorMessage = `HTTP Error ${response.status}: ${response.statusText}`;
    let details = null;

    try {
      const data = await response.json();
      if (data && data.detail) {
        if (typeof data.detail === 'string') {
          errorMessage = data.detail;
        } else if (Array.isArray(data.detail)) {
          // Pydantic validation error array
          errorMessage = data.detail.map(err => {
            const field = err.loc ? err.loc.join('.') : 'field';
            return `${field}: ${err.msg}`;
          }).join('; ');
          details = data.detail;
        } else {
          errorMessage = JSON.stringify(data.detail);
        }
      }
    } catch {
      // Body wasn't JSON or could not be parsed
    }

    throw new ApiError(errorMessage, response.status, details);
  }

  return response.json();
}

/**
 * Fetch system health and corpus statistics.
 */
export async function getHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError('Unable to connect to the backend server. Please verify the API is running.', 0);
  }
}

/**
 * Score a proposal (evaluate novelty).
 * Does NOT save the proposal to the database.
 * @param {Object} proposalData - 8 required sections + optional domain/project_id
 */
export async function scoreProposal(proposalData) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/score`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(proposalData),
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError('Network error while evaluating proposal.', 0);
  }
}

/**
 * Save an evaluated proposal into the SQLite database.
 * @param {Object} proposal - ProposalInput matching 8 sections
 * @param {Object} evaluationReport - Exact report object returned by scoreProposal
 */
export async function saveProposal(proposal, evaluationReport) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/save`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({
        proposal,
        evaluation_report: evaluationReport,
      }),
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError('Network error while saving proposal.', 0);
  }
}

/**
 * Fetch paginated list of submitted proposals.
 * @param {number} limit
 * @param {number} offset
 */
export async function getProposals(limit = 20, offset = 0) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/proposals?limit=${limit}&offset=${offset}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError('Network error while fetching proposals.', 0);
  }
}

/**
 * Fetch full proposal details and saved evaluation report by project ID.
 * @param {string} projectId
 */
export async function getProposalById(projectId) {
  try {
    const encodedId = encodeURIComponent(projectId);
    const res = await fetch(`${API_BASE_URL}/api/proposals/${encodedId}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });
    return await handleResponse(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(`Network error while fetching proposal "${projectId}".`, 0);
  }
}
