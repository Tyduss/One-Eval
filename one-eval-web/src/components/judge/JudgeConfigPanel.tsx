import React, { useState } from 'react';

interface ModelInfo {
  model_name: string;
  detail_path: string;
}

interface JudgeConfigPanelProps {
  apiBaseUrl: string;
  models: ModelInfo[];
  onStart: (config: {
    scoringPrompt: string;
    judgeModel: { model_name_or_path: string; api_url: string; api_key: string };
    concurrency: number;
    selectedModels: string[];
  }) => void;
  lang: 'zh' | 'en';
}

const DEFAULT_SCORING_PROMPT = `你是一位专业的AI评测裁判。

【评判流程】
1. 先仔细阅读用户的原始 Prompt，理解其需求意图
2. 在脑海中构建一个理想的满分回答应该是什么样的
3. 然后审阅模型实际输出，与理想回答对标
4. 按以下维度和尺度打分

【评分维度】
1. 推理过程(think)：0-5分
   - 5分：推理逻辑清晰完整，准确理解Prompt需求，推导链路严谨
   - 3分：有基本推理但存在明显缺陷或遗漏
   - 0分：无有效推理或推理完全错误
2. 回答质量(body)：0-5分
   - 5分：回答准确完整地满足了Prompt需求，表达清晰有条理
   - 3分：基本正确但不够完整或有明显瑕疵
   - 0分：完全错误或未作答

【输出要求】
请严格按以下JSON格式输出：
{
    "think_score": <0-5的数字>,
    "body_score": <0-5的数字>,
    "overall_score": <0-5的数字>,
    "critical_issue": "<最严重的一个问题，无则填'无'>",
    "other_issues": ["<问题1>", "<问题2>"],
    "remark": "<整体评价备注>"
}`;

export default function JudgeConfigPanel({ apiBaseUrl, models, onStart, lang }: JudgeConfigPanelProps) {
  const tt = (zh: string, en: string) => (lang === 'zh' ? zh : en);

  const [scoringPrompt, setScoringPrompt] = useState(DEFAULT_SCORING_PROMPT);
  const [judgeModelName, setJudgeModelName] = useState('');
  const [judgeApiUrl, setJudgeApiUrl] = useState('');
  const [judgeApiKey, setJudgeApiKey] = useState('');
  const [judgeApiProvider, setJudgeApiProvider] = useState('openai_compatible');
  const [concurrency, setConcurrency] = useState(5);
  const [selectedModels, setSelectedModels] = useState<string[]>(
    models.map(m => m.model_name)
  );

  // Load agent config as default judge model
  React.useEffect(() => {
    fetch(`${apiBaseUrl}/api/config/agent`)
      .then(r => r.json())
      .then(cfg => {
        setJudgeModelName(cfg.model || '');
        setJudgeApiUrl(cfg.base_url || '');
      })
      .catch(() => {});
  }, [apiBaseUrl]);

  const toggleModel = (name: string) => {
    setSelectedModels(prev =>
      prev.includes(name)
        ? prev.filter(n => n !== name)
        : [...prev, name]
    );
  };

  const handleStart = () => {
    if (!scoringPrompt.trim()) return;
    if (!judgeModelName.trim() || !judgeApiUrl.trim()) return;
    if (selectedModels.length === 0) return;

    onStart({
      scoringPrompt: scoringPrompt.trim(),
      judgeModel: {
        model_name_or_path: judgeModelName.trim(),
        api_url: judgeApiUrl.trim(),
        api_key: judgeApiKey.trim(),
        api_provider: judgeApiProvider,
      },
      concurrency,
      selectedModels,
    });
  };

  return (
    <div className="space-y-4">
      {/* Scoring Prompt */}
      <div>
        <label className="block text-xs font-semibold text-slate-600 mb-1">
          {tt('执行标准 Prompt', 'Scoring Standard Prompt')}
        </label>
        <textarea
          value={scoringPrompt}
          onChange={e => setScoringPrompt(e.target.value)}
          rows={10}
          className="w-full border border-slate-200 rounded-lg px-3 py-2 text-xs font-mono
                     focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent
                     resize-y bg-white"
          placeholder={tt('请输入评分标准...', 'Enter scoring criteria...')}
        />
      </div>

      {/* Judge Model Config */}
      <div className="grid grid-cols-3 gap-3">
        <div>
          <label className="block text-xs font-medium text-slate-500 mb-1">
            {tt('裁判模型', 'Judge Model')}
          </label>
          <input
            type="text"
            value={judgeModelName}
            onChange={e => setJudgeModelName(e.target.value)}
            className="w-full border border-slate-200 rounded-lg px-3 py-1.5 text-xs
                       focus:outline-none focus:ring-2 focus:ring-blue-400"
            placeholder="gpt-4o"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-500 mb-1">
            {tt('并发数', 'Concurrency')}
          </label>
          <input
            type="number"
            value={concurrency}
            onChange={e => setConcurrency(Math.max(1, parseInt(e.target.value) || 1))}
            min={1}
            max={20}
            className="w-full border border-slate-200 rounded-lg px-3 py-1.5 text-xs
                       focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-500 mb-1">
            {tt('裁判 API Key', 'Judge API Key')}
          </label>
          <input
            type="password"
            value={judgeApiKey}
            onChange={e => setJudgeApiKey(e.target.value)}
            className="w-full border border-slate-200 rounded-lg px-3 py-1.5 text-xs
                       focus:outline-none focus:ring-2 focus:ring-blue-400"
            placeholder={tt('留空用系统 Key', 'Leave empty')}
          />
        </div>
      </div>

      <div>
        <label className="block text-xs font-medium text-slate-500 mb-1">
          {tt('裁判 API URL', 'Judge API URL')}
        </label>
        <input
          type="text"
          value={judgeApiUrl}
          onChange={e => setJudgeApiUrl(e.target.value)}
          className="w-full border border-slate-200 rounded-lg px-3 py-1.5 text-xs
                     focus:outline-none focus:ring-2 focus:ring-blue-400"
          placeholder="https://api.openai.com/v1"
        />
      </div>

      <div>
        <label className="block text-xs font-medium text-slate-500 mb-1">
          {tt('裁判 API 协议', 'Judge API Protocol')}
        </label>
        <select
          title={tt('选择 API 协议', 'Select API Protocol')}
          value={judgeApiProvider}
          onChange={e => setJudgeApiProvider(e.target.value)}
          className="w-full border border-slate-200 rounded-lg px-3 py-1.5 text-xs
                     focus:outline-none focus:ring-2 focus:ring-blue-400"
        >
          <option value="openai_compatible">{tt('OpenAI 兼容', 'OpenAI Compatible')}</option>
          <option value="anthropic">Anthropic</option>
        </select>
      </div>

      {/* Model Selection */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs font-semibold text-slate-600">
            {tt('待评分模型', 'Models to Judge')}
          </label>
          <button
            onClick={() => setSelectedModels(
              selectedModels.length === models.length
                ? []
                : models.map(m => m.model_name)
            )}
            className="text-[10px] text-blue-500 hover:text-blue-700"
          >
            {selectedModels.length === models.length
              ? tt('取消全选', 'Deselect All')
              : tt('全选', 'Select All')}
          </button>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {models.map(m => (
            <button
              key={m.model_name}
              onClick={() => toggleModel(m.model_name)}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-medium border transition-colors ${
                selectedModels.includes(m.model_name)
                  ? 'bg-blue-50 border-blue-300 text-blue-700'
                  : 'bg-white border-slate-200 text-slate-500 hover:bg-slate-50'
              }`}
            >
              {m.model_name}
            </button>
          ))}
        </div>
      </div>

      {/* Estimate + Start */}
      <div className="flex items-center justify-between pt-2 border-t border-slate-100">
        <span className="text-[10px] text-amber-600">
          {tt(
            `预估: ${selectedModels.length} 个模型 × 样本数`,
            `Est: ${selectedModels.length} models × samples`
          )}
        </span>
        <button
          onClick={handleStart}
          disabled={!scoringPrompt.trim() || !judgeModelName.trim() || selectedModels.length === 0}
          className="px-4 py-1.5 rounded-lg text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700
                     disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
        >
          {tt('开始评分', 'Start Judging')}
        </button>
      </div>
    </div>
  );
}
