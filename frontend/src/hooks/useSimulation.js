import { useCallback, useEffect, useRef, useState } from 'react'

export const SENTIMENT = ['Normal', 'Accident', 'Roadwork', 'Weather']

const TWEET_POOL = [
  { text: 'Traffic is moving smoothly on the main corridor today.', label: 0 },
  { text: 'Morning commute looking clean, all lanes open.', label: 0 },
  { text: 'Green lights all the way downtown, love it.', label: 0 },
  { text: 'Major accident on the main road, avoid the area!', label: 1 },
  { text: 'Two-car collision at 5th & Grand, traffic backing up fast.', label: 1 },
  { text: 'Road construction is causing long delays.', label: 2 },
  { text: 'Lane closures for resurfacing work until Friday, plan ahead.', label: 2 },
  { text: 'Heavy rain and flooding reported at the intersection.', label: 3 },
  { text: 'Dense fog on the highway, visibility under 50 meters.', label: 3 },
]

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))
const rand = (lo, hi) => lo + Math.random() * (hi - lo)
let tweetId = 0

function perceive(density) {
  const cars = Math.round(density / 4 + rand(0, 3))
  return {
    cars,
    trucks: Math.round(density / 18 + rand(0, 1.4)),
    buses: Math.round(density / 30 + rand(0, 1.1)),
    motorcycles: Math.round(density / 22 + rand(0, 1.4)),
    pedestrians: Math.round(rand(0, 5) + density / 25),
  }
}

function ppoDecide([cur, pred, anomaly, sentiment]) {
  if (anomaly > 0.5) return 1
  if (sentiment === 1) return 1 // accident
  if (cur > 62 || pred > 72) return 1
  if (sentiment === 3 && cur > 45) return 1 // weather + moderate load
  return 0
}

function computeMockStep(prev, step) {
  const baseline = 42 + 18 * Math.sin(step / 7)
  const relief = prev.decision === 1 ? -7 : -1
  let density = clamp(
    prev.currentDensity + (baseline - prev.currentDensity) * 0.22 + relief + rand(-7, 10),
    4, 100,
  )
  const anomaly = Math.random() < 0.13 ? 1 : 0
  const reconError = anomaly ? rand(0.62, 0.97) : rand(0.04, 0.3)
  if (anomaly) density = clamp(density + rand(12, 26), 4, 100)
  const predicted = clamp(density + rand(-10, 10), 0, 100)
  const pool = anomaly && Math.random() < 0.7 ? TWEET_POOL.filter((t) => t.label !== 0) : TWEET_POOL
  const pick = pool[Math.floor(Math.random() * pool.length)]
  const tweet = {
    id: ++tweetId,
    text: pick.text,
    label: pick.label,
    confidence: rand(0.84, 0.99),
    time: new Date().toLocaleTimeString('en-GB', { hour12: false }),
  }
  const stateVector = [density, predicted, anomaly, pick.label]
  const decision = prev.override === 'green' ? 0 : prev.override === 'emergency' ? 1 : ppoDecide(stateVector)

  return {
    ...prev,
    step,
    currentDensity: density,
    predictedDensity: predicted,
    anomaly: anomaly === 1,
    reconError,
    counts: perceive(density),
    sentiment: pick.label,
    latestTweet: tweet,
    tweets: [tweet, ...prev.tweets].slice(0, 24),
    decision,
    confidence: rand(0.87, 0.995),
    stateVector,
  }
}

const initialState = {
  step: 0,
  currentDensity: 32,
  predictedDensity: 36,
  anomaly: false,
  reconError: 0.12,
  counts: { cars: 8, trucks: 2, buses: 1, motorcycles: 1, pedestrians: 3 },
  sentiment: 0,
  latestTweet: null,
  tweets: [],
  decision: 0,
  confidence: 0.94,
  stateVector: [32, 36, 0, 0],
  override: null, // null | 'green' | 'emergency' | 'locked'
}

export function useSimulation(tickMs = 3000) {
  const [running, setRunning] = useState(true)
  const [state, setState] = useState(initialState)
  const overrideRef = useRef(null)
  
  const [history, setHistory] = useState(() =>
    Array.from({ length: 24 }, (_, i) => ({ t: i - 24, cur: 30 + Math.sin(i / 3) * 6, pred: 33 + Math.sin(i / 3 + 1) * 6 })),
  )
  const stepRef = useRef(0)

  const advance = useCallback(async () => {
    if (overrideRef.current === 'locked') return

    try {
      // 1. Try to fetch from the live Python Backend (ONLY IF RUNNING LOCALLY)
      if (import.meta.env.DEV) {
        const overrideQuery = overrideRef.current ? `?override=${overrideRef.current}` : ''
        
        // Set a short timeout so it fails quickly if backend isn't there
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 1500);
        
        const res = await fetch(`http://localhost:8000/api/state${overrideQuery}`, { signal: controller.signal })
        clearTimeout(timeoutId);
        
        if (!res.ok) throw new Error('API Response not OK');
        
        const next = await res.json()
        
        setState((prev) => {
          if (prev.override === 'locked') return prev
          stepRef.current += 1
          
          const nextState = {
            ...prev,
            step: stepRef.current,
            currentDensity: next.currentDensity,
            predictedDensity: next.predictedDensity,
            anomaly: next.anomaly,
            reconError: next.reconError,
            counts: next.counts,
            sentiment: next.sentiment,
            latestTweet: next.latestTweet,
            tweets: next.latestTweet ? [next.latestTweet, ...prev.tweets].slice(0, 24) : prev.tweets,
            decision: next.decision,
            confidence: next.confidence,
            stateVector: next.stateVector,
          }
          
          setHistory((h) => [...h.slice(-39), { t: nextState.step, cur: nextState.currentDensity, pred: nextState.predictedDensity }])
          return nextState
        })
        return; // Exit if successful
      } else {
        throw new Error('Production Mode: Skipping localhost fetch');
      }
      
    } catch (err) {
      // 2. FALLBACK: If the Python server is offline (like when deployed on Vercel), use Mock Data!
      console.warn('API unreachable, using Mock Simulation Data Fallback.')
      
      setState((prev) => {
        if (prev.override === 'locked') return prev
        stepRef.current += 1
        
        const nextState = computeMockStep(prev, stepRef.current)
        setHistory((h) => [...h.slice(-39), { t: nextState.step, cur: nextState.currentDensity, pred: nextState.predictedDensity }])
        return nextState
      })
    }
  }, [])

  useEffect(() => {
    if (!running) return undefined
    const id = setInterval(advance, tickMs)
    return () => clearInterval(id)
  }, [running, advance, tickMs])

  const start = useCallback(() => setRunning(true), [])
  const pause = useCallback(() => setRunning(false), [])

  const stepForward = useCallback(() => {
    setRunning(false)
    advance()
  }, [advance])

  const reset = useCallback(() => {
    stepRef.current = 0
    setRunning(false)
    setState(initialState)
    setHistory(Array.from({ length: 24 }, (_, i) => ({ t: i - 24, cur: 30 + Math.sin(i / 3) * 6, pred: 33 + Math.sin(i / 3 + 1) * 6 })))
  }, [])

  const setOverride = useCallback((mode) => {
    setState((prev) => {
      const newOverride = prev.override === mode ? null : mode
      overrideRef.current = newOverride
      let decision = prev.decision
      if (newOverride === 'green') decision = 0
      if (newOverride === 'emergency') decision = 1
      return { ...prev, override: newOverride, decision }
    })
  }, [])

  return { state, history, running, start, pause, stepForward, reset, setOverride }
}
