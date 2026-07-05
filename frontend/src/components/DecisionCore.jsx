import { motion, AnimatePresence } from 'framer-motion'
import { SENTIMENT } from '../hooks/useSimulation'

function Ring({ size, color, dur, reverse, dash }) {
  return (
    <motion.div
      className="core-ring"
      style={{
        width: size,
        height: size,
        borderColor: 'transparent',
        borderTopColor: color,
        borderRightColor: dash ? 'transparent' : color,
        opacity: 0.6,
        boxShadow: `0 0 14px ${color}33`,
      }}
      animate={{ rotate: reverse ? -360 : 360 }}
      transition={{ duration: dur, repeat: Infinity, ease: 'linear' }}
    />
  )
}

export default function DecisionCore({ decision, confidence, override, stateVector }) {
  const locked = override === 'locked'
  const emergency = decision === 1 && !locked

  const ringColor = locked ? '#ffb020' : emergency ? '#ff2d55' : '#3dff8f'
  const action = locked
    ? 'INTERSECTION LOCKED'
    : emergency
      ? 'EMERGENCY CLEARANCE'
      : 'GREEN — NORMAL FLOW'
  const actionClass = locked ? 'alock' : emergency ? 'a1' : 'a0'

  const [cur, pred, anom, sent] = stateVector

  return (
    <motion.section
      className="panel"
      initial={{ y: 24, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.7, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="panel-head">
        <div className="panel-title"><span className="dot c" /> RL Decision Core — PPO</div>
        <div className="panel-sub">POLICY π(a|s)</div>
      </div>

      <div className="core">
        <div className="core-stage">
          {/* ambient glow that tracks the decision */}
          <motion.div
            style={{
              position: 'absolute', width: 240, height: 240, borderRadius: '50%',
              pointerEvents: 'none', filter: 'blur(34px)',
            }}
            animate={{ background: `radial-gradient(circle, ${ringColor}30 0%, transparent 65%)`, scale: [1, 1.12, 1] }}
            transition={{ scale: { duration: 2.4, repeat: Infinity, ease: 'easeInOut' }, background: { duration: 0.8 } }}
          />
          <Ring size={196} color={ringColor} dur={9} />
          <Ring size={166} color={ringColor} dur={6} reverse dash />
          <Ring size={230} color="#00e5ff" dur={16} dash />

          <motion.div
            className="tlight"
            animate={{ rotateY: [-9, 9, -9] }}
            transition={{ duration: 7, repeat: Infinity, ease: 'easeInOut' }}
          >
            <motion.div
              className={`lamp red${locked ? ' on' : ''}`}
              animate={locked ? { opacity: [1, 0.55, 1] } : { opacity: 1 }}
              transition={{ duration: 1.2, repeat: locked ? Infinity : 0 }}
            />
            <motion.div
              className={`lamp amber${emergency ? ' on' : ''}`}
              animate={emergency ? { opacity: [1, 0.45, 1] } : { opacity: 1 }}
              transition={{ duration: 0.8, repeat: emergency ? Infinity : 0 }}
            />
            <motion.div
              className={`lamp green${!locked ? ' on' : ''}`}
              animate={emergency && !locked ? { opacity: [1, 0.5, 1] } : { opacity: 1 }}
              transition={{ duration: 0.8, repeat: emergency ? Infinity : 0 }}
            />
          </motion.div>
        </div>

        <div className="core-readout">
          <AnimatePresence mode="wait">
            <motion.div
              key={action}
              className={`core-action ${actionClass}`}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.35 }}
            >
              {action}
            </motion.div>
          </AnimatePresence>
          <div className="core-conf">
            {override && override !== 'locked'
              ? `MANUAL OVERRIDE ENGAGED — POLICY BYPASSED`
              : `ACTION ${decision} · POLICY CONFIDENCE ${(confidence * 100).toFixed(1)}%`}
          </div>
        </div>

        <div className="state-vector">
          <div className="sv-cell"><div className="v">{cur.toFixed(0)}</div><div className="k">Density</div></div>
          <div className="sv-cell"><div className="v">{pred.toFixed(0)}</div><div className="k">Pred</div></div>
          <div className="sv-cell">
            <div className="v" style={anom ? { color: '#ff2d55' } : undefined}>{anom}</div>
            <div className="k">Anomaly</div>
          </div>
          <div className="sv-cell"><div className="v">{sent}</div><div className="k">{SENTIMENT[sent]}</div></div>
        </div>
      </div>
    </motion.section>
  )
}
