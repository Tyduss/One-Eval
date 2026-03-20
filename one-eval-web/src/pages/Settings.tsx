import { useMemo, useState, useEffect } from "react";
import axios from "axios";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Plus, Save, Database, Cloud, KeyRound, Trash2, PlugZap,
  ChevronDown, CheckCircle2, Pencil, X
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useLang } from "@/lib/i18n";

interface ModelConfig {
  name: string;
  path: string;
  is_api?: boolean;
  api_url?: string;
  api_key?: string;
}

interface SettingsCardProps {
  title: string;
  description: string;
  icon: React.ElementType;
  iconColorClass?: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

const SettingsCard = ({ 
  title, 
  description, 
  icon: Icon, 
  iconColorClass = "bg-primary/10 text-primary", 
  children, 
  defaultOpen = false 
}: SettingsCardProps) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <Card className="overflow-hidden border-slate-200 shadow-sm hover:shadow-md transition-all duration-300">
      <CardHeader 
        className="cursor-pointer bg-slate-50/30 hover:bg-slate-50/80 transition-colors p-6"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={`p-2.5 rounded-xl ${iconColorClass}`}>
              <Icon className="w-6 h-6" />
            </div>
            <div>
              <CardTitle className="text-lg font-semibold text-slate-900">{title}</CardTitle>
              <CardDescription className="text-slate-500 mt-1">{description}</CardDescription>
            </div>
          </div>
          <ChevronDown 
            className={`w-5 h-5 text-slate-400 transition-transform duration-300 ${isOpen ? "rotate-180" : ""}`} 
          />
        </div>
      </CardHeader>
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
          >
            <div className="border-t border-slate-100">
              <CardContent className="p-6 pt-6 space-y-6">
                {children}
              </CardContent>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
};

export const Settings = () => {
  const { lang } = useLang();
  const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [selectedModelIdxs, setSelectedModelIdxs] = useState<Set<number>>(new Set());
  const [newModel, setNewModel] = useState<ModelConfig>({ name: "", path: "", is_api: false, api_url: "", api_key: "" });
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editModel, setEditModel] = useState<ModelConfig>({ name: "", path: "", is_api: false, api_url: "", api_key: "" });
  const [loading, setLoading] = useState(false);
  const [apiBaseUrl] = useState(() => localStorage.getItem("oneEval.apiBaseUrl") || "http://localhost:8000");
  const [hfEndpoint, setHfEndpoint] = useState("https://hf-mirror.com");
  const [hfToken, setHfToken] = useState("");
  const [hfTokenSet, setHfTokenSet] = useState(false);
  const [savingHf, setSavingHf] = useState(false);
  const [agentBaseUrl, setAgentBaseUrl] = useState("http://123.129.219.111:3000/v1");
  const [agentModel, setAgentModel] = useState("gpt-4o");
  const [agentApiKeyInput, setAgentApiKeyInput] = useState("");
  const [agentApiKeySet, setAgentApiKeySet] = useState(false);
  const [agentTimeoutS, setAgentTimeoutS] = useState(15);
  const [savingAgent, setSavingAgent] = useState(false);
  const [testingAgent, setTestingAgent] = useState(false);
  const [agentTestResult, setAgentTestResult] = useState<string | null>(null);
  const [showAgentSuccess, setShowAgentSuccess] = useState(false);
  const [testingModelPath, setTestingModelPath] = useState<string | null>(null);
  const [modelTestMsg, setModelTestMsg] = useState<Record<string, string>>({});

  const agentUrlPresets = useMemo(
    () => [
      { label: "yuchaAPI", value: "http://123.129.219.111:3000/v1/chat/completions" },
      { label: "OpenAI", value: "https://api.openai.com/v1" },
      { label: "OpenRouter", value: "https://openrouter.ai/api/v1" },
      { label: "Apiyi (OpenAI Compatible)", value: "https://api.apiyi.com/v1" },
      { label: "Custom...", value: "__custom__" },
    ],
    []
  );
  const agentUrlPresetValue = useMemo(() => {
    const hit = agentUrlPresets.find((p) => p.value === agentBaseUrl);
    return hit ? hit.value : "__custom__";
  }, [agentUrlPresets, agentBaseUrl]);

  const isValidHttpUrl = (u: string) => {
    try {
      const parsed = new URL(u);
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch {
      return false;
    }
  };

  useEffect(() => {
    if (!isValidHttpUrl(apiBaseUrl)) return;
    fetchModels();
    fetchHfConfig();
    fetchAgentConfig();
  }, [apiBaseUrl]);

  const fetchModels = async () => {
    try {
      const res = await axios.get(`${apiBaseUrl}/api/models`);
      setModels(res.data);
      // 恢复选中的模型
      try {
        const saved = localStorage.getItem("oneEval.selectedModels");
        if (saved) {
          const idxs = JSON.parse(saved);
          setSelectedModelIdxs(new Set(idxs));
        }
      } catch (e) {}
    } catch (e) {
      console.error("Failed to fetch models", e);
    }
  };

  const handleSaveModel = async () => {
    if (!newModel.name || !newModel.path) return;
    setLoading(true);
    try {
      await axios.post(`${apiBaseUrl}/api/models`, newModel);
      setModels([...models, newModel]);
      setNewModel({ name: "", path: "" });
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const fetchHfConfig = async () => {
    try {
      const res = await axios.get(`${apiBaseUrl}/api/config/hf`);
      setHfEndpoint(res.data.endpoint || "https://hf-mirror.com");
      setHfTokenSet(Boolean(res.data.token_set));
    } catch (e) {
      setHfEndpoint("https://hf-mirror.com");
      setHfTokenSet(false);
    }
  };

  const handleSaveHfConfig = async () => {
    setSavingHf(true);
    try {
      const payload: any = { endpoint: hfEndpoint };
      if (hfToken.trim()) payload.token = hfToken;
      const res = await axios.post(`${apiBaseUrl}/api/config/hf`, payload);
      setHfEndpoint(res.data.endpoint || hfEndpoint);
      setHfTokenSet(Boolean(res.data.token_set));
      setHfToken("");
    } catch (e) {
      console.error(e);
    }
    setSavingHf(false);
  };

  const handleClearHfToken = async () => {
    setSavingHf(true);
    try {
      const res = await axios.post(`${apiBaseUrl}/api/config/hf`, { clear_token: true });
      setHfEndpoint(res.data.endpoint || hfEndpoint);
      setHfTokenSet(Boolean(res.data.token_set));
      setHfToken("");
    } catch (e) {
      console.error(e);
    }
    setSavingHf(false);
  };

  const fetchAgentConfig = async () => {
    try {
      const res = await axios.get(`${apiBaseUrl}/api/config/agent`);
      setAgentBaseUrl(res.data.base_url || "http://123.129.219.111:3000/v1");
      setAgentModel(res.data.model || "gpt-4o");
      setAgentApiKeySet(Boolean(res.data.api_key_set));
      setAgentTimeoutS(Number(res.data.timeout_s || 15));
      setAgentApiKeyInput("");
    } catch (e) {
      setAgentBaseUrl("http://123.129.219.111:3000/v1");
      setAgentModel("gpt-4o");
      setAgentApiKeySet(false);
      setAgentTimeoutS(15);
      setAgentApiKeyInput("");
    }
  };

  const handleSaveAgentConfig = async () => {
    setSavingAgent(true);
    try {
      const payload: any = {
        base_url: agentBaseUrl,
        model: agentModel,
        timeout_s: agentTimeoutS,
      };
      if (agentApiKeyInput.trim()) payload.api_key = agentApiKeyInput.trim();
      const res = await axios.post(`${apiBaseUrl}/api/config/agent`, payload);
      setAgentBaseUrl(res.data.base_url || agentBaseUrl);
      setAgentModel(res.data.model || agentModel);
      setAgentApiKeySet(Boolean(res.data.api_key_set));
      setAgentTimeoutS(Number(res.data.timeout_s || agentTimeoutS));
      // Keep the input and result visible so user knows what happened
      // setAgentApiKeyInput(""); 
      // setAgentTestResult(null);
      setShowAgentSuccess(true);
      setTimeout(() => setShowAgentSuccess(false), 3000);
    } catch (e) {
      console.error(e);
    }
    setSavingAgent(false);
  };

  const handleClearAgentApiKey = async () => {
    setSavingAgent(true);
    try {
      const res = await axios.post(`${apiBaseUrl}/api/config/agent`, { clear_api_key: true });
      setAgentApiKeySet(Boolean(res.data.api_key_set));
      setAgentApiKeyInput("");
      setAgentTestResult(null);
    } catch (e) {
      console.error(e);
    }
    setSavingAgent(false);
  };

  const handleTestAgentConnection = async () => {
    setTestingAgent(true);
    setAgentTestResult(null);
    try {
      const payload: any = {
        base_url: agentBaseUrl,
        model: agentModel,
        timeout_s: agentTimeoutS,
      };
      // Send the currently input API key if it's not empty, otherwise don't send it (let backend use saved key)
      // Actually, if we want to test "what I just typed", we should send it even if empty string?
      // But if user has a saved key and clears the input, maybe they mean "use saved"?
      // No, consistent UX: "Test" tests the *current form values*.
      // If user clears the input, they might mean "no auth".
      // However, for security, we don't auto-fill the input with the saved key.
      // So if input is empty, and there IS a saved key (agentApiKeySet is true), we probably want to use the saved key.
      // If input is NOT empty, use the input.
      if (agentApiKeyInput.trim()) {
        payload.api_key = agentApiKeyInput.trim();
      } else if (!agentApiKeySet) {
          // No saved key, and no input key -> send empty to override any potential default? 
          // Backend falls back to saved config if req.api_key is None.
          // If we send "", backend treats it as empty key.
          // If we don't send it, backend uses saved key.
          // If there is NO saved key (agentApiKeySet false), backend has None.
          // So if input is empty and not saved, we can just send nothing.
      } else {
         // Input empty, but key is saved. 
         // We should NOT send api_key field so backend uses the saved one.
      }

      const res = await axios.post(`${apiBaseUrl}/api/config/agent/test`, payload);
      if (res.data.ok) {
        setAgentTestResult(`OK (${res.data.mode})`);
      } else {
        const code = res.data.status_code ? ` [${res.data.status_code}]` : "";
        setAgentTestResult(`${tt("失败", "FAILED")}${code}: ${res.data.detail}`);
      }
    } catch (e) {
      setAgentTestResult(`${tt("失败", "FAILED")}: ${tt("请求异常", "request error")}`);
    }
    setTestingAgent(false);
  };

  const getModelTestKey = (model: ModelConfig) => {
    if (model.is_api) {
      return `api:${model.api_url}:${model.path}`;
    }
    return `local:${model.path}`;
  };

  const canTestModel = (model: ModelConfig) => {
    if (model.is_api) {
      return (model.api_url || "").trim() && (model.path || "").trim();
    }
    return (model.path || "").trim();
  };

  const canSaveModel = (model: ModelConfig) => {
    return (model.name || "").trim() && canTestModel(model);
  };

  const handleTestModel = async (model: ModelConfig) => {
    const key = getModelTestKey(model);
    setTestingModelPath(key);
    setModelTestMsg((prev) => ({ ...prev, [key]: tt("测试中...", "Testing...") }));
    try {
      const res = await axios.post(`${apiBaseUrl}/api/models/test`, {
        is_api: model.is_api,
        path: model.path,
        api_url: model.api_url,
        api_key: model.api_key,
      });
      const ok = !!res.data?.ok;
      setModelTestMsg((prev) => ({
        ...prev,
        [key]: ok ? tt("连接成功", "Connection OK") : `${tt("失败", "Failed")}: ${res.data?.detail || ""}`,
      }));
    } catch (e: any) {
      const detail = e?.response?.data?.detail || tt("请求异常", "request error");
      setModelTestMsg((prev) => ({ ...prev, [key]: `${tt("失败", "Failed")}: ${detail}` }));
    }
    setTestingModelPath(null);
  };

  const handleDeleteModel = async (index: number) => {
    try {
      await axios.delete(`${apiBaseUrl}/api/models/${index}`);
      setModels(models.filter((_, i) => i !== index));
    } catch (e) {
      console.error(e);
    }
  };

  const handleUpdateModel = async (index: number) => {
    if (!editModel.name || !editModel.path) return;
    try {
      // 如果 api_key 为空，保留原来的 key
      const modelToUpdate = { ...editModel };
      if (!modelToUpdate.api_key && models[index]?.api_key) {
        modelToUpdate.api_key = models[index].api_key;
      }
      // 更新后端
      await axios.put(`${apiBaseUrl}/api/models/${index}`, modelToUpdate);
      // 更新本地状态
      const newModels = [...models];
      newModels[index] = modelToUpdate;
      setModels(newModels);
      setEditingIndex(null);
      setEditModel({ name: "", path: "", is_api: false, api_url: "", api_key: "" });
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="p-12 max-w-[1600px] mx-auto space-y-8">
      <div className="space-y-2 mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">{tt("设置", "Settings")}</h1>
        <p className="text-slate-500 text-lg">{tt("配置评测环境与模型注册表。", "Configure your evaluation environment and model registry.")}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 items-start">
        {/* 1. One-Eval Backend (Hidden) */}

        {/* 2. Agent Server */}
        <SettingsCard
          title={tt("Agent 服务", "Agent Server")}
          description={tt("配置用于评测流程的 LLM 提供方（如 OpenAI、vLLM 等）。", "Configure the LLM provider (e.g. OpenAI, vLLM, etc.) used for evaluation.")}
          icon={PlugZap}
          iconColorClass="bg-violet-500/10 text-violet-600"
          defaultOpen={true}
        >
          <div className="space-y-6">
            <div className="space-y-2">
              <Label>{tt("服务地址（必填）", "Provider URL (Required)")}</Label>
              <div className="grid grid-cols-1 gap-2">
                <select
                  value={agentUrlPresetValue}
                  onChange={(e) => {
                    const v = e.target.value;
                    if (v !== "__custom__") setAgentBaseUrl(v);
                  }}
                  className="w-full h-10 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                >
                  {agentUrlPresets.map((p) => (
                    <option key={p.value} value={p.value}>
                      {p.label}
                    </option>
                  ))}
                </select>
                <Input
                  value={agentBaseUrl}
                  onChange={(e) => setAgentBaseUrl(e.target.value)}
                  placeholder={tt("例如：https://api.openai.com/v1", "e.g. https://api.openai.com/v1")}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>{tt("模型（必填）", "Model (Required)")}</Label>
                <Input
                  value={agentModel}
                  onChange={(e) => setAgentModel(e.target.value)}
                  placeholder="gpt-4o / deepseek-v3 / ..."
                  list="agent-model-presets"
                  className="border-slate-200"
                />
                <datalist id="agent-model-presets">
                  <option value="gpt-4o" />
                  <option value="gpt-5.1" />
                  <option value="gpt-5.2" />
                  <option value="deepseek-v3" />
                  <option value="deepseek-r1" />
                </datalist>
              </div>
              <div className="space-y-2">
                <Label>{tt("超时秒数（可选）", "Timeout in seconds (Optional)")}</Label>
                <Input
                  type="number"
                  value={agentTimeoutS}
                  onChange={(e) => setAgentTimeoutS(Number(e.target.value || 15))}
                  placeholder="15"
                  className="border-slate-200"
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>{tt("API Key（可选）", "API Key (Optional)")}</Label>
                {agentApiKeySet && <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">{tt("密钥已保存", "Key Saved")}</span>}
              </div>
              <Input
                type="password"
                value={agentApiKeyInput}
                onChange={(e) => setAgentApiKeyInput(e.target.value)}
                placeholder="sk-..."
              />
            </div>

            <div className="flex gap-3">
              <Button variant="outline" className="flex-1" onClick={handleTestAgentConnection} disabled={testingAgent}>
                {testingAgent ? tt("测试中...", "Testing...") : tt("测试连接", "Test Connection")}
              </Button>
              <Button
                className={`flex-1 text-white transition-all duration-300 ${
                  showAgentSuccess 
                    ? "bg-emerald-600 hover:bg-emerald-700 shadow-emerald-600/20" 
                    : "bg-slate-900 hover:bg-slate-800"
                }`}
                onClick={handleSaveAgentConfig}
                disabled={savingAgent}
              >
                {savingAgent ? (
                  tt("保存中...", "Saving...")
                ) : showAgentSuccess ? (
                  <><CheckCircle2 className="w-4 h-4 mr-2" /> {tt("已保存", "Saved!")}</>
                ) : (
                  tt("保存配置", "Save Configuration")
                )}
              </Button>
            </div>

            <div className="flex items-center justify-between pt-2">
               <Button variant="ghost" size="sm" className="text-red-500 hover:text-red-600 hover:bg-red-50" onClick={handleClearAgentApiKey} disabled={savingAgent}>
                <Trash2 className="w-4 h-4 mr-2" />
                {tt("清除 API Key", "Clear API Key")}
              </Button>
              {agentTestResult && (
                <div className={`text-xs px-3 py-1.5 rounded-md font-mono ${agentTestResult.startsWith("OK") ? "bg-emerald-50 text-emerald-700 border border-emerald-200" : "bg-red-50 text-red-700 border border-red-200"}`}>
                  {agentTestResult}
                </div>
              )}
            </div>
          </div>
        </SettingsCard>

        {/* 3. HuggingFace */}
        <SettingsCard
          title={tt("HuggingFace 配置", "HuggingFace Configuration")}
          description={tt("配置 HF 镜像地址和访问令牌，用于下载模型与数据集。", "Configure HF mirror endpoint and access token for downloading models/datasets.")}
          icon={Cloud}
          iconColorClass="bg-amber-500/10 text-amber-600"
          defaultOpen={false}
        >
          <div className="space-y-6">
            <div className="space-y-2">
              <Label>{tt("HF 镜像地址", "HF Mirror Endpoint")}</Label>
              <Input
                value={hfEndpoint}
                onChange={(e) => setHfEndpoint(e.target.value)}
                placeholder="https://hf-mirror.com"
              />
              <p className="text-xs text-slate-500">{tt("默认值：https://hf-mirror.com", "Default: https://hf-mirror.com")}</p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>HF Token</Label>
                {hfTokenSet && <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">{tt("令牌已保存", "Token Saved")}</span>}
              </div>
              <Input
                type="password"
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
                placeholder="hf_..."
              />
              <p className="text-xs text-slate-500">
                {tt("留空则保留当前已保存令牌。", "Leave empty to keep the currently saved token.")}
              </p>
            </div>

            <div className="flex gap-3">
              <Button
                className="flex-1 text-white bg-slate-900 hover:bg-slate-800"
                onClick={handleSaveHfConfig}
                disabled={savingHf}
              >
                {savingHf ? tt("保存中...", "Saving...") : <><KeyRound className="w-4 h-4 mr-2" /> {tt("保存 HF 配置", "Save HF Config")}</>}
              </Button>
              <Button
                variant="outline"
                className="flex-1 text-red-500 hover:text-red-600 hover:bg-red-50"
                onClick={handleClearHfToken}
                disabled={savingHf}
              >
                <Trash2 className="w-4 h-4 mr-2" />
                {tt("清除令牌", "Clear Token")}
              </Button>
            </div>
          </div>
        </SettingsCard>

        {/* 4. Model Registry */}
        <SettingsCard
          title={tt("目标模型注册表", "Target Model Registry")}
          description={tt("注册你希望参与评测的本地或远端模型。", "Register local or remote models that you want to evaluate.")}
          icon={Database}
          iconColorClass="bg-pink-500/10 text-pink-600"
          defaultOpen={true}
        >
          <div className="space-y-6">
            {/* Add New */}
            <div className="p-5 border border-slate-200 rounded-xl bg-slate-50/50 space-y-4">
              <h4 className="text-sm font-semibold flex items-center gap-2 text-slate-800">
                <Plus className="w-4 h-4" /> {tt("新增模型", "Add New Model")}
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>{tt("模型名称（必填）", "Model Name (Required)")}</Label>
                  <Input
                    placeholder={tt("例如：Qwen2.5-7B-Instruct", "e.g. Qwen2.5-7B-Instruct")}
                    value={newModel.name}
                    onChange={e => setNewModel({...newModel, name: e.target.value})}
                    className="bg-white"
                  />
                </div>
                <div className="space-y-2">
                  <Label>{tt("模型类型", "Model Type")}</Label>
                  <select
                    value={newModel.is_api ? "api" : "local"}
                    onChange={e => setNewModel({...newModel, is_api: e.target.value === "api"})}
                    className="w-full h-10 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                  >
                    <option value="local">{tt("本地模型 / HuggingFace", "Local / HuggingFace")}</option>
                    <option value="api">{tt("API 服务", "API Service")}</option>
                  </select>
                </div>
              </div>

              {/* 本地模型配置 */}
              {!newModel.is_api && (
                <div className="space-y-2">
                  <Label>{tt("模型路径（必填）", "Model Path (Required)")}</Label>
                  <Input
                    placeholder="/mnt/models/... 或 Qwen/Qwen2.5-7B-Instruct"
                    value={newModel.path}
                    onChange={e => setNewModel({...newModel, path: e.target.value})}
                    className="bg-white"
                  />
                </div>
              )}

              {/* API 模型配置 */}
              {newModel.is_api && (
                <div className="space-y-4 p-4 border border-violet-200 rounded-lg bg-violet-50/50">
                  <div className="space-y-2">
                    <Label>{tt("API 地址（必填）", "API URL (Required)")}</Label>
                    <Input
                      placeholder="http://your-server:8080/v1"
                      value={newModel.api_url || ""}
                      onChange={e => setNewModel({...newModel, api_url: e.target.value})}
                      className="bg-white"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>{tt("模型标识（必填）", "Model Identifier (Required)")}</Label>
                    <Input
                      placeholder="gpt-4o / deepseek-v3 / ..."
                      value={newModel.path}
                      onChange={e => setNewModel({...newModel, path: e.target.value})}
                      className="bg-white"
                    />
                    <p className="text-xs text-slate-500">{tt("API 服务中该模型的名称标识", "The model identifier in your API service")}</p>
                  </div>
                  <div className="space-y-2">
                    <Label>{tt("API Key（可选）", "API Key (Optional)")}</Label>
                    <Input
                      type="password"
                      placeholder="sk-..."
                      value={newModel.api_key || ""}
                      onChange={e => setNewModel({...newModel, api_key: e.target.value})}
                      className="bg-white"
                    />
                  </div>
                </div>
              )}

              <div className="grid grid-cols-2 gap-3">
                <Button
                  variant="outline"
                  onClick={() => handleTestModel(newModel)}
                  disabled={!canTestModel(newModel) || !!testingModelPath}
                  className="w-full"
                >
                  {testingModelPath === getModelTestKey(newModel) ? tt("测试中...", "Testing...") : tt("测试连接", "Test Connection")}
                </Button>
                <Button onClick={handleSaveModel} disabled={loading || !canSaveModel(newModel)} className="w-full text-white bg-slate-900 hover:bg-slate-800">
                  {loading ? tt("保存中...", "Saving...") : <><Save className="w-4 h-4 mr-2"/> {tt("加入注册表", "Add to Registry")}</>}
                </Button>
              </div>
              {!canSaveModel(newModel) && (
                <div className="text-xs text-amber-600 bg-amber-50 px-3 py-2 rounded border border-amber-200">
                  {tt(
                    `请填写：${!newModel.name ? "模型名称" : ""}${newModel.is_api ? (!newModel.api_url ? "、API 地址" : "") + (!newModel.path ? "、模型标识" : "") : (!newModel.path ? "、模型路径" : "")}`,
                    `Please fill: ${!newModel.name ? "Model Name" : ""}${newModel.is_api ? (!newModel.api_url ? ", API URL" : "") + (!newModel.path ? ", Model ID" : "") : (!newModel.path ? ", Model Path" : "")}`
                  )}
                </div>
              )}
              {getModelTestKey(newModel) && modelTestMsg[getModelTestKey(newModel)] && (
                <div className={`text-xs px-3 py-2 rounded border ${modelTestMsg[getModelTestKey(newModel)].includes(tt("成功", "OK")) ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-red-50 text-red-700 border-red-200"}`}>
                  {modelTestMsg[getModelTestKey(newModel)]}
                </div>
              )}
            </div>

            {/* List */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium text-slate-500 uppercase tracking-wider text-xs">{tt("已注册模型", "Registered Models")}</h4>
                {models.length > 0 && (
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <span>{tt("勾选参与评测", "Select for evaluation")}</span>
                    <span className="text-emerald-600 font-medium">({selectedModelIdxs.size}/{models.length})</span>
                  </div>
                )}
              </div>
              {models.length === 0 && (
                <div className="text-center py-8 border-2 border-dashed border-slate-200 rounded-xl">
                  <p className="text-sm text-slate-400">{tt("暂未注册模型。", "No models registered yet.")}</p>
                </div>
              )}
              <div className="grid grid-cols-1 gap-3">
                {models.map((m, i) => (
                  <div key={i}>
                    {editingIndex === i ? (
                      /* 编辑模式 */
                      <div className="p-4 rounded-xl border border-blue-200 bg-blue-50/30 space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label>{tt("模型名称（必填）", "Model Name (Required)")}</Label>
                            <Input
                              value={editModel.name}
                              onChange={e => setEditModel({...editModel, name: e.target.value})}
                              className="bg-white"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label>{tt("模型类型", "Model Type")}</Label>
                            <select
                              title={tt("选择模型类型", "Select model type")}
                              value={editModel.is_api ? "api" : "local"}
                              onChange={e => setEditModel({...editModel, is_api: e.target.value === "api"})}
                              className="w-full h-10 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900"
                            >
                              <option value="local">{tt("本地模型 / HuggingFace", "Local / HuggingFace")}</option>
                              <option value="api">{tt("API 服务", "API Service")}</option>
                            </select>
                          </div>
                        </div>
                        {!editModel.is_api ? (
                          <div className="space-y-2">
                            <Label>{tt("模型路径（必填）", "Model Path (Required)")}</Label>
                            <Input
                              value={editModel.path}
                              onChange={e => setEditModel({...editModel, path: e.target.value})}
                              className="bg-white"
                            />
                          </div>
                        ) : (
                          <div className="space-y-4 p-4 border border-violet-200 rounded-lg bg-violet-50/50">
                            <div className="space-y-2">
                              <Label>{tt("API 地址（必填）", "API URL (Required)")}</Label>
                              <Input
                                value={editModel.api_url || ""}
                                onChange={e => setEditModel({...editModel, api_url: e.target.value})}
                                className="bg-white"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label>{tt("模型标识（必填）", "Model Identifier (Required)")}</Label>
                              <Input
                                value={editModel.path}
                                onChange={e => setEditModel({...editModel, path: e.target.value})}
                                className="bg-white"
                              />
                            </div>
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <Label>{tt("API Key", "API Key")}</Label>
                                {models[i]?.api_key && !editModel.api_key && (
                                  <span className="text-[10px] text-amber-600">{tt("原密钥已保留，留空则保持不变", "Original key preserved, leave empty to keep")}</span>
                                )}
                              </div>
                              <Input
                                type="text"
                                placeholder={models[i]?.api_key ? tt("留空保持原密钥", "Leave empty to keep original") : tt("sk-... (可选)", "sk-... (optional)")}
                                value={editModel.api_key || ""}
                                onChange={e => setEditModel({...editModel, api_key: e.target.value})}
                                className="bg-white"
                              />
                            </div>
                          </div>
                        )}
                        <div className="flex gap-2">
                          <Button
                            onClick={() => handleUpdateModel(i)}
                            disabled={!editModel.name || !editModel.path}
                            className="flex-1 bg-blue-600 hover:bg-blue-700 text-white"
                          >
                            <Save className="w-4 h-4 mr-2" /> {tt("保存修改", "Save Changes")}
                          </Button>
                          <Button
                            variant="outline"
                            onClick={() => { setEditingIndex(null); setEditModel({ name: "", path: "", is_api: false, api_url: "", api_key: "" }); }}
                            className="flex-1"
                          >
                            <X className="w-4 h-4 mr-2" /> {tt("取消", "Cancel")}
                          </Button>
                        </div>
                      </div>
                    ) : (
                      /* 查看模式 */
                      <div className={`flex items-center justify-between p-4 rounded-xl border transition-all ${selectedModelIdxs.has(i) ? "border-emerald-300 bg-emerald-50/30" : "border-slate-100 bg-white hover:border-slate-200"}`}>
                        <div className="flex items-center gap-3 flex-1 min-w-0 mr-4">
                          <input
                            type="checkbox"
                            title={tt("选择参与评测", "Select for evaluation")}
                            checked={selectedModelIdxs.has(i)}
                            onChange={(e) => {
                              const newSet = new Set(selectedModelIdxs);
                              if (e.target.checked) {
                                newSet.add(i);
                              } else {
                                newSet.delete(i);
                              }
                              setSelectedModelIdxs(newSet);
                              localStorage.setItem("oneEval.selectedModels", JSON.stringify([...newSet]));
                            }}
                            className="w-4 h-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                          />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="font-semibold text-slate-900">{m.name}</span>
                              {m.is_api && <span className="text-xs px-2 py-0.5 rounded-full bg-violet-100 text-violet-700">API</span>}
                            </div>
                            <div className="text-xs text-slate-500 truncate font-mono mt-1" title={m.is_api ? m.api_url : m.path}>
                              {m.is_api ? `${m.api_url} → ${m.path}` : m.path}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-slate-500 hover:text-slate-700 hover:bg-slate-100"
                            onClick={() => {
                              setEditingIndex(i);
                              // 编辑时不带 api_key，防止泄露
                              setEditModel({ ...m, api_key: "" });
                            }}
                          >
                            <Pencil className="w-4 h-4" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleTestModel(m)}
                            disabled={testingModelPath === getModelTestKey(m)}
                          >
                            {testingModelPath === getModelTestKey(m) ? tt("测试中...", "Testing...") : tt("测试", "Test")}
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-red-500 hover:text-red-600 hover:bg-red-50"
                            onClick={() => handleDeleteModel(i)}
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                    )}
                    {modelTestMsg[getModelTestKey(m)] && (
                      <div className={`-mt-2 mb-1 text-xs px-3 py-2 rounded border ${modelTestMsg[getModelTestKey(m)].includes(tt("成功", "OK")) ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-red-50 text-red-700 border-red-200"}`}>
                        {modelTestMsg[getModelTestKey(m)]}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </SettingsCard>

      </div>
    </div>
  );
};

export default Settings;
