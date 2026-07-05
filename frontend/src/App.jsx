import { useSimulation } from './hooks/useSimulation'
import Header from './components/Header'
import ControlDeck from './components/ControlDeck'
import VisionPanel from './components/VisionPanel'
import DensityChart from './components/DensityChart'
import AnomalyAlert from './components/AnomalyAlert'
import NLPFeed from './components/NLPFeed'
import DecisionCore from './components/DecisionCore'
import PipelineStatus from './components/PipelineStatus'

export default function App() {
  const { state, history, running, start, pause, stepForward, reset, setOverride } = useSimulation()

  return (
    <div className="stage">
      <Header running={running} override={state.override} step={state.step} />

      <div className="grid">
        <div className="col">
          <VisionPanel counts={state.counts} density={state.currentDensity} step={state.step} />
          <PipelineStatus running={running && state.override !== 'locked'} step={state.step} />
        </div>

        <div className="col">
          <AnomalyAlert anomaly={state.anomaly} reconError={state.reconError} />
          <DensityChart history={history} current={state.currentDensity} predicted={state.predictedDensity} />
          <NLPFeed tweets={state.tweets} />
        </div>

        <div className="col right">
          <DecisionCore
            decision={state.decision}
            confidence={state.confidence}
            override={state.override}
            stateVector={state.stateVector}
          />
          <ControlDeck
            running={running}
            override={state.override}
            start={start}
            pause={pause}
            stepForward={stepForward}
            reset={reset}
            setOverride={setOverride}
          />
        </div>
      </div>
    </div>
  )
}
