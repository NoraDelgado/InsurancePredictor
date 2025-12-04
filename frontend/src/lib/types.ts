export interface InsuranceInput {
  age: number
  gender: 'male' | 'female'
  bmi: number
  bloodpressure: number
  diabetic: 'Yes' | 'No'
  children: number
  smoker: 'Yes' | 'No'
  region: 'northeast' | 'northwest' | 'southeast' | 'southwest'
}

export interface PredictionResult {
  predicted_charge: number
  model_version: string
}

export interface ApiError {
  error: string
  detail: string
}
