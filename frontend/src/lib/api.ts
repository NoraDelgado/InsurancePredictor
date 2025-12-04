import { InsuranceInput, PredictionResult } from './types'

// In production, use the Render.com hosted backend
// In development, use localhost
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 
  (typeof window !== 'undefined' && window.location.hostname !== 'localhost' 
    ? 'https://insurance-predictor-api.onrender.com'  // Your Render backend URL
    : 'http://localhost:8000')

export async function predictInsuranceCost(
  input: InsuranceInput
): Promise<PredictionResult> {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(input),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to get prediction')
  }

  return response.json()
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`)
    const data = await response.json()
    return data.status === 'healthy'
  } catch {
    return false
  }
}

