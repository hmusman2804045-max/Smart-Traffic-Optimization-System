import { motion } from 'framer-motion'

const MODELS = [
  { name: 'PERCEPTION', model: 'YOLOv8n · vision', color: '#00e5ff' },
  { name: 'FORECASTER', model: 'LSTM · time-series', color: '#ff3df2' },
  { name: 'ANOMALY', model: 'VAE · reconstruction', color: '#ffb020' },
  { name: 'SENTIMENT', model: 'BERT-tiny · 4-class', color: '#8b7bff' },
  { name: 'OPTIMIZER', model: 'PPO · stable-baselines3', color: '#3dff8f' },
]

export default function PipelineStatus({ running, step }) {
  return (
    <motion.section
      className="panel"
      initial={{ y: 24, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.7, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="panel-head">
        <div className="panel-title"><span className="dot c" /> AI Pipeline</div>
        <div className="panel-sub">5 MODELS · CUDA:0</div>
      </div>
      <div className="panel-body">
        <div className="pipeline">
          {MODELS.map((m, i) => {
            // deterministic pseudo-load per model, drifts with step
            const load = 22 + ((step * 7 + i * 13) % 31) + i * 6
            return (
              <div className="pipe" key={m.name}>
                <motion.span
                  style={{ width: 8, height: 8, borderRadius: '50%', background: m.color, boxShadow: `0 0 10px ${m.color}`, flex: 'none' }}
                  animate={running ? { opacity: [1, 0.3, 1] } : { opacity: 0.45 }}
                  transition={{ duration: 1.5 + i * 0.22, repeat: Infinity }}
                />
                <div>
                  <div className="pipe-name">{m.name}</div>
                  <div className="pipe-model">{m.model}</div>
                </div>
                <div className="pipe-load">{running ? `${load}ms` : 'IDLE'}</div>
              </div>
            )
          })}
        </div>
      </div>
    </motion.section>
  )
}
