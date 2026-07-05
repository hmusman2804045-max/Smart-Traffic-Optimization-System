import { motion, AnimatePresence } from 'framer-motion'

const W = 560
const H = 190
const PAD = { top: 12, right: 10, bottom: 18, left: 30 }

function pathFrom(points, key) {
  const iw = W - PAD.left - PAD.right
  const ih = H - PAD.top - PAD.bottom
  const n = points.length
  const coords = points.map((p, i) => [
    PAD.left + (i / Math.max(1, n - 1)) * iw,
    PAD.top + (1 - p[key] / 100) * ih,
  ])
  // smooth with quadratic midpoints
  let d = `M ${coords[0][0]},${coords[0][1]}`
  for (let i = 1; i < coords.length; i++) {
    const [px, py] = coords[i - 1]
    const [cx, cy] = coords[i]
    const mx = (px + cx) / 2
    d += ` Q ${px},${py} ${mx},${(py + cy) / 2}`
  }
  d += ` L ${coords[n - 1][0]},${coords[n - 1][1]}`
  return { d, last: coords[n - 1] }
}

export default function DensityChart({ history, current, predicted }) {
  const cur = pathFrom(history, 'cur')
  const pred = pathFrom(history, 'pred')
  const areaD = `${cur.d} L ${W - PAD.right},${H - PAD.bottom} L ${PAD.left},${H - PAD.bottom} Z`

  return (
    <motion.section
      className="panel"
      initial={{ y: 24, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.7, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="panel-head">
        <div className="panel-title"><span className="dot c" /> Density Telemetry — LSTM Forecast</div>
        <div className="legend">
          <span><i style={{ background: '#00e5ff', boxShadow: '0 0 8px #00e5ff' }} /> CURRENT</span>
          <span><i style={{ background: '#ff3df2', boxShadow: '0 0 8px #ff3df2' }} /> PREDICTED T+1H</span>
        </div>
      </div>
      <div className="panel-body">
        <div className="metric-row">
          <div className="metric cyan">
            <div className="k">Current Density</div>
            <AnimatePresence mode="popLayout">
              <motion.div className="v" key={Math.round(current)}
                initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
                {current.toFixed(1)}<span className="u">/100</span>
              </motion.div>
            </AnimatePresence>
          </div>
          <div className="metric magenta">
            <div className="k">Predicted T+1H</div>
            <AnimatePresence mode="popLayout">
              <motion.div className="v" key={Math.round(predicted)}
                initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
                {predicted.toFixed(1)}<span className="u">/100</span>
              </motion.div>
            </AnimatePresence>
          </div>
          <div className="metric green">
            <div className="k">Trend Δ</div>
            <div className="v" style={predicted - current < 0 ? undefined : { color: '#ffb020', textShadow: '0 0 16px rgba(255,176,32,0.5)' }}>
              {predicted - current >= 0 ? '+' : ''}{(predicted - current).toFixed(1)}
            </div>
          </div>
          <div className="metric violet">
            <div className="k">Grid Load</div>
            <div className="v">{Math.round(current * 0.9 + 8)}<span className="u">%</span></div>
          </div>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }}>
          <defs>
            <filter id="glowCyan" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="3.2" result="b" />
              <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <filter id="glowMag" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="3.2" result="b" />
              <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <linearGradient id="areaCyan" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stopColor="#00e5ff" stopOpacity="0.22" />
              <stop offset="1" stopColor="#00e5ff" stopOpacity="0" />
            </linearGradient>
          </defs>

          {/* gridlines */}
          {[0, 25, 50, 75, 100].map((v) => {
            const y = PAD.top + (1 - v / 100) * (H - PAD.top - PAD.bottom)
            return (
              <g key={v}>
                <line x1={PAD.left} x2={W - PAD.right} y1={y} y2={y} stroke="rgba(94,234,255,0.08)" strokeDasharray="3 5" />
                <text x={PAD.left - 6} y={y + 3} fill="#43516a" fontSize="8.5" fontFamily="Share Tech Mono" textAnchor="end">{v}</text>
              </g>
            )
          })}
          {/* congestion threshold */}
          <line x1={PAD.left} x2={W - PAD.right}
            y1={PAD.top + (1 - 0.62) * (H - PAD.top - PAD.bottom)}
            y2={PAD.top + (1 - 0.62) * (H - PAD.top - PAD.bottom)}
            stroke="rgba(255,45,85,0.4)" strokeDasharray="6 6" strokeWidth="1" />
          <text x={W - PAD.right} y={PAD.top + (1 - 0.62) * (H - PAD.top - PAD.bottom) - 5}
            fill="rgba(255,45,85,0.7)" fontSize="8" fontFamily="Share Tech Mono" textAnchor="end">CONGESTION THRESHOLD</text>

          <path d={areaD} fill="url(#areaCyan)" />
          <path d={pred.d} fill="none" stroke="#ff3df2" strokeWidth="1.8" strokeDasharray="6 5" filter="url(#glowMag)" opacity="0.9" />
          <path d={cur.d} fill="none" stroke="#00e5ff" strokeWidth="2.2" filter="url(#glowCyan)" />

          {/* live head dots */}
          <circle cx={cur.last[0]} cy={cur.last[1]} r="3.4" fill="#00e5ff" filter="url(#glowCyan)">
            <animate attributeName="r" values="3;5;3" dur="1.6s" repeatCount="indefinite" />
          </circle>
          <circle cx={pred.last[0]} cy={pred.last[1]} r="2.8" fill="#ff3df2" filter="url(#glowMag)" />
        </svg>
      </div>
    </motion.section>
  )
}
