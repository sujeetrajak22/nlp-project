import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error)
)

// Response interceptor
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred'
    throw new Error(message)
  }
)

/**
 * Predict news category
 * @param {string} headline - News headline
 * @param {string} article - News article text
 * @param {string} modelType - 'tfidf_lr' | 'distilbert'
 */
export const predictNews = async (headline, article, modelType = 'tfidf_lr') => {
  return api.post('/predict', { headline, article, model_type: modelType })
}

/**
 * Run preprocessing pipeline on text
 * @param {string} text - Raw input text
 */
export const preprocessText = async (text) => {
  return api.post('/preprocess', { text })
}

/**
 * Get dataset statistics
 */
export const getDatasetStats = async () => {
  return api.get('/dataset/stats')
}

/**
 * Get example news articles
 */
export const getExamples = async () => {
  return api.get('/examples')
}

/**
 * Get model performance metrics
 */
export const getModelMetrics = async () => {
  return api.get('/model/metrics')
}

/**
 * Health check
 */
export const getHealth = async () => {
  return api.get('/health')
}

/**
 * Get supported categories
 */
export const getCategories = async () => {
  return api.get('/categories')
}

export default api
