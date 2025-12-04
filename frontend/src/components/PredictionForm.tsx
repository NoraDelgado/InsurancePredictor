'use client'

import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { Loader2 } from 'lucide-react'
import { InsuranceInput, PredictionResult } from '@/lib/types'
import { predictInsuranceCost } from '@/lib/api'

const formSchema = z.object({
  age: z.number().min(18, 'Age must be at least 18').max(100, 'Age must be at most 100'),
  gender: z.enum(['male', 'female']),
  bmi: z.number().min(10, 'BMI must be at least 10').max(60, 'BMI must be at most 60'),
  bloodpressure: z.number().min(60, 'Blood pressure must be at least 60').max(200, 'Blood pressure must be at most 200'),
  diabetic: z.enum(['Yes', 'No']),
  children: z.number().min(0, 'Children must be 0 or more').max(10, 'Children must be at most 10'),
  smoker: z.enum(['Yes', 'No']),
  region: z.enum(['northeast', 'northwest', 'southeast', 'southwest']),
})

interface PredictionFormProps {
  onResult: (result: PredictionResult) => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

export function PredictionForm({ onResult, isLoading, setIsLoading }: PredictionFormProps) {
  const [error, setError] = useState<string | null>(null)

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<InsuranceInput>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      age: 30,
      gender: 'male',
      bmi: 25.0,
      bloodpressure: 80,
      diabetic: 'No',
      children: 0,
      smoker: 'No',
      region: 'southeast',
    },
  })

  async function onSubmit(data: InsuranceInput) {
    setIsLoading(true)
    setError(null)

    try {
      const result = await predictInsuranceCost(data)
      onResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Age */}
        <div>
          <label htmlFor="age" className="form-label">
            Age
          </label>
          <input
            id="age"
            type="number"
            {...register('age', { valueAsNumber: true })}
            className="form-input"
            placeholder="Enter your age"
          />
          {errors.age && (
            <p className="mt-1 text-sm text-red-400">{errors.age.message}</p>
          )}
        </div>

        {/* Gender */}
        <div>
          <label htmlFor="gender" className="form-label">
            Gender
          </label>
          <select
            id="gender"
            {...register('gender')}
            className="form-select"
          >
            <option value="male">Male</option>
            <option value="female">Female</option>
          </select>
        </div>

        {/* BMI */}
        <div>
          <label htmlFor="bmi" className="form-label">
            BMI
          </label>
          <input
            id="bmi"
            type="number"
            step="0.1"
            {...register('bmi', { valueAsNumber: true })}
            className="form-input"
            placeholder="Enter your BMI"
          />
          {errors.bmi && (
            <p className="mt-1 text-sm text-red-400">{errors.bmi.message}</p>
          )}
        </div>

        {/* Blood Pressure */}
        <div>
          <label htmlFor="bloodpressure" className="form-label">
            Blood Pressure
          </label>
          <input
            id="bloodpressure"
            type="number"
            {...register('bloodpressure', { valueAsNumber: true })}
            className="form-input"
            placeholder="Enter blood pressure"
          />
          {errors.bloodpressure && (
            <p className="mt-1 text-sm text-red-400">{errors.bloodpressure.message}</p>
          )}
        </div>

        {/* Diabetic */}
        <div>
          <label htmlFor="diabetic" className="form-label">
            Diabetic
          </label>
          <select
            id="diabetic"
            {...register('diabetic')}
            className="form-select"
          >
            <option value="No">No</option>
            <option value="Yes">Yes</option>
          </select>
        </div>

        {/* Children */}
        <div>
          <label htmlFor="children" className="form-label">
            Number of Children
          </label>
          <input
            id="children"
            type="number"
            {...register('children', { valueAsNumber: true })}
            className="form-input"
            placeholder="Number of children"
          />
          {errors.children && (
            <p className="mt-1 text-sm text-red-400">{errors.children.message}</p>
          )}
        </div>

        {/* Smoker */}
        <div>
          <label htmlFor="smoker" className="form-label">
            Smoker
          </label>
          <select
            id="smoker"
            {...register('smoker')}
            className="form-select"
          >
            <option value="No">No</option>
            <option value="Yes">Yes</option>
          </select>
        </div>

        {/* Region */}
        <div>
          <label htmlFor="region" className="form-label">
            Region
          </label>
          <select
            id="region"
            {...register('region')}
            className="form-select"
          >
            <option value="northeast">Northeast</option>
            <option value="northwest">Northwest</option>
            <option value="southeast">Southeast</option>
            <option value="southwest">Southwest</option>
          </select>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20">
          <p className="text-red-400">{error}</p>
        </div>
      )}

      <button
        type="submit"
        disabled={isLoading}
        className="btn-primary flex items-center justify-center gap-2"
      >
        {isLoading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Calculating...
          </>
        ) : (
          'Get Prediction'
        )}
      </button>
    </form>
  )
}
