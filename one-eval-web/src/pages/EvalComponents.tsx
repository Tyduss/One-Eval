import React, { useRef, useEffect, useState, Component, type ErrorInfo, type ReactNode } from "react";
import { motion, AnimatePresence } from "framer-motion";
import JudgeConfigPanel from "@/components/judge/JudgeConfigPanel";
import JudgeProgressPanel from "@/components/judge/JudgeProgressPanel";
import {
    Send, Check, Loader2, AlertCircle, ChevronDown, ChevronUp,
    Database, Bot, Maximize2, Minimize2, X, Save as SaveIcon, Tag, Upload,
    ClipboardCheck, BarChart3, History, Eye, Download, RotateCw
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type { Lang } from "@/lib/i18n";


import { ReportView } from "./ReportComponents";
import { SimpleMarkdown } from "@/components/ui/simple-markdown";

// --- Types ---
export interface Bench {
  bench_name: string;
  eval_type?: string;
  bench_dataflow_eval_type?: string;
  meta?: any;
  eval_status?: string;
  download_status?: string;
  dataset_cache?: string;
}

export interface WorkflowState {
  user_query: string;
  benches: Bench[];
  target_model_name?: string;
  target_model?: any;
  metric_plan?: Record<string, any[]>;
  reports?: Record<string, any>;
}

// per-bench × per-model 持久化镜像（由 workflow_meta_store 写入，
// SSE snapshot/get_status 返回）。失败模型也会在里面，方便前端展示
// 红色 badge + 重试按钮。
export interface WorkflowMetaModel {
  status?: "pending" | "running" | "success" | "failed" | "cancelled";
  stage?: string;
  generated?: number;
  total?: number;
  percent?: number;
  error?: string | null;
  started_at?: string;
  finished_at?: string;
  // 批次级失败计数（_BatchStatus 实时投影）。
  consecutive_failures?: number;
  max_failures?: number;
  failures?: number;
  successes?: number;
  last_error?: string | null;
}
export interface WorkflowMetaBench {
  models?: Record<string, WorkflowMetaModel>;
}
export interface WorkflowMetaSnapshot {
  thread_id?: string;
  status?: string;
  started_at?: string;
  updated_at?: string;
  cancel_reason?: string | null;
  benches?: Record<string, WorkflowMetaBench>;
}

// Defensive boundary so a broken bench / model entry can't crash an entire tab.
// （Plan §C.3：Tab 2 白屏的兜底防御。）
interface TabErrorBoundaryState { hasError: boolean; message?: string }
export class TabErrorBoundary extends Component<{ children: ReactNode; fallbackLabel?: string }, TabErrorBoundaryState> {
    state: TabErrorBoundaryState = { hasError: false };
    static getDerivedStateFromError(err: Error): TabErrorBoundaryState {
        return { hasError: true, message: err?.message || "render error" };
    }
    componentDidCatch(error: Error, info: ErrorInfo) {
        // eslint-disable-next-line no-console
        console.error("[TabErrorBoundary]", error, info?.componentStack);
    }
    render() {
        if (this.state.hasError) {
            return (
                <div className="flex flex-col items-center justify-center py-12 text-slate-400 gap-2">
                    <AlertCircle className="w-8 h-8 text-amber-400" />
                    <p className="text-sm">{this.props.fallbackLabel || "该面板出错，请刷新或切换其他 Tab"}</p>
                    {this.state.message && (
                        <p className="text-[10px] text-slate-300 font-mono">{this.state.message}</p>
                    )}
                </div>
            );
        }
        return this.props.children;
    }
}

// Per-model status pill. Used in SummaryBenchCard table headers and Tab 2 entries.
export const ModelStatusBadge = ({ status, lang }: { status?: string; lang: Lang }) => {
    const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
    if (!status) return null;
    const cfg: Record<string, { cls: string; label: string }> = {
        running: { cls: "bg-blue-50 text-blue-600 border-blue-100", label: tt("运行中", "Running") },
        success: { cls: "bg-emerald-50 text-emerald-700 border-emerald-100", label: tt("成功", "Success") },
        failed: { cls: "bg-red-50 text-red-700 border-red-100", label: tt("失败", "Failed") },
        cancelled: { cls: "bg-amber-50 text-amber-700 border-amber-100", label: tt("已取消", "Cancelled") },
        pending: { cls: "bg-slate-50 text-slate-500 border-slate-100", label: tt("待运行", "Pending") },
    };
    const it = cfg[status] || { cls: "bg-slate-50 text-slate-500 border-slate-100", label: status };
    return (
        <span className={cn("text-[9px] font-bold px-1.5 py-0.5 rounded border whitespace-nowrap", it.cls)}>
            {it.label}
        </span>
    );
};

// Indicate consecutive failure counter for a running model. Shows "失败 N/M" or
// "fail N/M" with tooltip on last error. When N=0 and last_error is null, returns null.
export const FailureCounterBadge = ({
    consecutive,
    max,
    lastError,
    lang,
}: {
    consecutive?: number;
    max?: number;
    lastError?: string | null;
    lang: Lang;
}) => {
    const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
    if (consecutive === undefined || consecutive === null) return null;
    if (consecutive <= 0 && !lastError) return null;
    const m = max || 3;
    const ratio = consecutive / m;
    // color escalates from amber → red as ratio approaches 1.
    const cls =
        ratio >= 1
            ? "bg-red-100 text-red-700 border-red-200"
            : ratio >= 0.5
                ? "bg-amber-50 text-amber-700 border-amber-200"
                : "bg-slate-50 text-slate-500 border-slate-200";
    const label = `${tt("失败", "fail")} ${consecutive}/${m}`;
    return (
        <span
            className={cn("text-[9px] font-mono px-1.5 py-0.5 rounded border whitespace-nowrap", cls)}
            title={lastError || undefined}
        >
            {label}
        </span>
    );
};

// --- Modal Component ---
const Modal = ({ isOpen, onClose, title, description, children, footer }: { isOpen: boolean, onClose: () => void, title: React.ReactNode, description?: string, children: React.ReactNode, footer?: React.ReactNode }) => {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 bg-black/40 backdrop-blur-sm"
                onClick={onClose}
            />
            <motion.div 
                initial={{ opacity: 0, scale: 0.95, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: 10 }}
                className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl max-h-[85vh] flex flex-col relative z-10 overflow-hidden"
            >
                <div className="p-6 border-b border-slate-100 flex items-start justify-between bg-slate-50/50">
                    <div>
                        <h3 className="text-xl font-bold text-slate-900 flex items-center gap-2">{title}</h3>
                        {description && <p className="text-sm text-slate-500 mt-1">{description}</p>}
                    </div>
                    <Button variant="ghost" size="icon" onClick={onClose} className="h-8 w-8 -mr-2 -mt-2">
                        <X className="w-5 h-5 text-slate-400" />
                    </Button>
                </div>
                <div className="flex-1 overflow-y-auto p-6">
                    {children}
                </div>
                {footer && (
                    <div className="p-4 border-t border-slate-100 bg-slate-50/50 flex justify-end gap-2">
                        {footer}
                    </div>
                )}
            </motion.div>
        </div>
    );
};

// --- Bench Card Component ---
export const BenchCard = ({ bench, activeNode, lang, onUpdate, onRetryDownload }: { bench: Bench, activeNode: string | null, lang: Lang, onUpdate?: (updatedBench: Bench) => void, onRetryDownload?: (params: { bench_name: string, config?: string, split?: string }) => Promise<void> }) => {
    const [isDetailsOpen, setIsDetailsOpen] = useState(false);
    const [isRetrying, setIsRetrying] = useState(false);
    const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
    const EVAL_TYPE_OPTIONS = [
        { value: "key1_text_score", label: tt("文本评分", "Text Scoring"), desc: tt("仅需文本字段", "Only text field required") },
        { value: "key2_qa", label: tt("问答单答案", "QA Single Answer"), desc: tt("问题+单参考答案", "Question + single reference") },
        { value: "key2_q_ma", label: tt("问答多答案", "QA Multi Answer"), desc: tt("问题+多参考答案", "Question + multiple references") },
        { value: "key3_q_choices_a", label: tt("单选题", "Multiple Choice"), desc: tt("题目+选项+正确选项", "Question + options + correct option") },
        { value: "key3_q_choices_as", label: tt("多选题", "Multi-select"), desc: tt("题目+选项+正确选项集合", "Question + options + labels set") },
        { value: "key3_q_a_rejected", label: tt("偏好对比", "Preference Pair"), desc: tt("题目+优选答案+拒绝答案", "Question + chosen + rejected") },
    ];
    const EVAL_TYPE_REQUIRED_KEYS: Record<string, string[]> = {
        key1_text_score: ["input_text_key"],
        key2_qa: ["input_question_key", "input_target_key"],
        key2_q_ma: ["input_question_key", "input_targets_key"],
        key3_q_choices_a: ["input_question_key", "input_choices_key", "input_label_key"],
        key3_q_choices_as: ["input_question_key", "input_choices_key", "input_labels_key"],
        key3_q_a_rejected: ["input_question_key", "input_better_key", "input_rejected_key"],
    };
    const EVAL_TYPE_GUIDE: Record<string, { scene: string; example: string }> = {
        key1_text_score: {
            scene: tt("纯文本质量评估（如困惑度）", "Pure text quality evaluation (e.g., perplexity)"),
            example: "text",
        },
        key2_qa: {
            scene: tt("问答任务，每题一个标准答案", "QA task with one reference answer"),
            example: "question, target",
        },
        key2_q_ma: {
            scene: tt("问答任务，每题多个可接受答案", "QA task with multiple acceptable answers"),
            example: "question, targets",
        },
        key3_q_choices_a: {
            scene: tt("单选题任务", "Single-choice task"),
            example: "question, choices, label",
        },
        key3_q_choices_as: {
            scene: tt("多选题任务", "Multi-select task"),
            example: "question, choices, labels",
        },
        key3_q_a_rejected: {
            scene: tt("偏好对比（好答案 vs 差答案）", "Preference pair (chosen vs rejected)"),
            example: "question, better, rejected",
        },
    };
    
    // Local state for editing in modal
    const [editKeyMap, setEditKeyMap] = useState<Record<string, string>>({});
    const [selectedSubset, setSelectedSubset] = useState<string>("");
    const [selectedSplit, setSelectedSplit] = useState<string>("");
    const [selectedEvalType, setSelectedEvalType] = useState<string>("");
    const [selectedPreviewRow, setSelectedPreviewRow] = useState<number>(0);
    
    // Safe parsing helper
    const safeParse = (data: any) => {
        if (!data) return null;
        if (typeof data === 'string') {
            try { return JSON.parse(data); } catch { return data; }
        }
        return data;
    };

    // Helper to check if value is a plain object (not array, not null)
    const isObject = (val: any) => val != null && typeof val === 'object' && !Array.isArray(val);
    
    const toLabel = (val: any): string => {
        if (val == null) return "";
        if (typeof val === "string") return val;
        if (typeof val === "number" || typeof val === "boolean") return String(val);
        if (typeof val === "object") {
            const name = (val as any).name;
            if (typeof name === "string") return name;
            try { return JSON.stringify(val); } catch { return String(val); }
        }
        return String(val);
    };
    const collapseMiddle = (raw: any, maxLen = 220, head = 120, tail = 80): string => {
        const text = typeof raw === "string" ? raw : JSON.stringify(raw);
        if (!text) return "";
        if (text.length <= maxLen) return text;
        return `${text.slice(0, head)} ... ${text.slice(-tail)}`;
    };

    // Extract Meta Data safely
    const meta = bench.meta || {};
    const structure = safeParse(meta.structure); 
    const keyMapping = safeParse(meta.key_mapping); 
    const downloadConfig = safeParse(meta.download_config); 
    const evalType = bench.bench_dataflow_eval_type || bench.eval_type || meta.bench_dataflow_eval_type; 
    const evalTypeLabel = (() => {
        const found = EVAL_TYPE_OPTIONS.find(x => x.value === (selectedEvalType || evalType));
        return found ? found.label : "";
    })();
    const descriptionRaw = meta.card_text ?? (lang === 'zh' ? (meta.description_zh || meta.description) : meta.description) ?? meta.desc ?? "No description available.";
    const description = typeof descriptionRaw === 'string' ? descriptionRaw : (descriptionRaw ? JSON.stringify(descriptionRaw) : "No description available.");
    const tags = Array.isArray(meta.tags) ? meta.tags : [];
    const availableKeys = Array.from(new Set([...(Array.isArray(meta.keys) ? meta.keys : []), ...(Array.isArray((bench as any).bench_keys) ? (bench as any).bench_keys : [])].map((x: any) => String(x))));
    const previewData = Array.isArray(meta.preview_data) ? meta.preview_data : []; // Preview rows
    const downloadPath = meta.download_path || meta.local_path || bench.dataset_cache;
    const sampleCountRaw = downloadConfig?.count || structure?.count || meta.count;
    const sampleCount = (() => {
        if (typeof sampleCountRaw === 'number') return sampleCountRaw;
        if (typeof sampleCountRaw === 'string') return sampleCountRaw;
        if (sampleCountRaw && typeof sampleCountRaw === 'object') {
            const num = (sampleCountRaw as any).num_examples ?? (sampleCountRaw as any).count;
            if (typeof num === 'number' || typeof num === 'string') return num;
            return JSON.stringify(sampleCountRaw);
        }
        return undefined;
    })();

    // Parse structure for selector
    const structureSubsetsRaw = Array.isArray(structure?.subsets) ? structure.subsets : [];
    const structureSubsets = structureSubsetsRaw
        .map((s: any) => {
            const subsetVal = s?.subset ?? s?.name ?? s;
            const subset = toLabel(subsetVal);
            const splitsRaw = s?.splits ?? [];
            const splits = Array.isArray(splitsRaw) ? splitsRaw.map(toLabel).filter(Boolean) : [];
            return { subset, splits };
        })
        .filter((s: any) => s?.subset);
    const keyMappingEntries = (() => {
        const required = EVAL_TYPE_REQUIRED_KEYS[selectedEvalType || String(evalType || "")] || [];
        const keys = Array.from(new Set([...required, ...Object.keys(editKeyMap || {})]));
        return keys;
    })();
    const keyOptionsListId = `key-options-${String(bench.bench_name || "bench").replace(/[^a-zA-Z0-9_-]/g, "_")}`;

    // Init state from bench
    useEffect(() => {
        if (!isDetailsOpen) return;
        if (keyMapping) {
            setEditKeyMap({ ...keyMapping });
        }
        setSelectedEvalType(typeof evalType === "string" ? evalType : "");
        if (downloadConfig) {
            setSelectedSubset(toLabel(downloadConfig.config) || "default");
            setSelectedSplit(toLabel(downloadConfig.split) || "test");
        } else if (structureSubsets.length > 0) {
            // Default selection if no config exists
            const first = structureSubsets[0];
            setSelectedSubset(toLabel(first.subset));
            if (first.splits && first.splits.length > 0) {
                setSelectedSplit(toLabel(first.splits[0]));
            }
        }
        setSelectedPreviewRow(0);
    }, [isDetailsOpen, bench.bench_name]);

    const handleSave = () => {
        if (!onUpdate) return;
        
        const updatedBench = { ...bench };
        if (!updatedBench.meta) updatedBench.meta = {};
        
        // Update Key Map
        updatedBench.meta.key_mapping = editKeyMap;
        updatedBench.bench_dataflow_eval_type = selectedEvalType || updatedBench.bench_dataflow_eval_type;
        updatedBench.meta.bench_dataflow_eval_type = selectedEvalType || updatedBench.meta.bench_dataflow_eval_type;
        
        // Update Config
        const currentReason = updatedBench.meta.download_config?.reason || "User manually selected configuration.";
        
        updatedBench.meta.download_config = {
             config: selectedSubset,
             split: selectedSplit,
             reason: currentReason
        };

        onUpdate(updatedBench);
        setIsDetailsOpen(false);
    };

    const downloadError = typeof meta?.download_error === "string" ? meta.download_error : (meta?.download_error ? JSON.stringify(meta.download_error) : "");

    // Determine status color
    const statusColor = bench.download_status === "success" ? "bg-green-50 text-green-600" : "bg-slate-50 text-slate-400";
    
    // Check if we have meaningful data to show
    const hasData = structure || keyMapping || downloadConfig || evalType || (meta && Object.keys(meta).length > 0);
    const structureReady = !!structure?.ok;
    const structureFailed = !!meta?.structure_error;
    const downloadReady = bench.download_status === "success";
    const downloadFailed = bench.download_status === "failed";
    
    // Determine loading state based on active node
    const isLoading = !hasData && (
        activeNode === "DatasetStructureNode" || 
        activeNode === "BenchConfigRecommendNode" || 
        activeNode === "BenchTaskInferNode"
    );

    return (
        <>
            <div 
                onClick={() => hasData && setIsDetailsOpen(true)}
                className={cn(
                    "p-4 bg-white rounded-xl border border-slate-100 shadow-sm transition-all relative h-full flex flex-col group",
                    hasData ? "hover:shadow-md cursor-pointer hover:border-blue-200" : "cursor-default"
                )}
            >
                <div className="flex justify-between mb-2 shrink-0 gap-2">
                    <div className="flex flex-col gap-1 min-w-0">
                        <span className="text-sm font-bold text-slate-700 truncate pr-2" title={bench.bench_name}>{bench.bench_name}</span>
                        {meta.from_gallery === true ? (
                            <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-600 border border-emerald-100 w-fit">
                                ✓ Gallery · Ready to run
                            </span>
                        ) : meta.from_gallery === false ? (
                            <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-amber-50 text-amber-600 border border-amber-100 w-fit">
                                ⚠ HF Search · May need config
                            </span>
                        ) : null}
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                        {bench.download_status === "failed" && onRetryDownload && (
                            <Button
                                size="sm"
                                variant="outline"
                                className="h-6 px-2 text-[10px] gap-1"
                                disabled={isRetrying}
                                onClick={async (e) => {
                                    e.stopPropagation();
                                    setIsRetrying(true);
                                    try {
                                        const dl = bench.meta?.download_config || {};
                                        await onRetryDownload({ bench_name: bench.bench_name, config: dl.config, split: dl.split });
                                    } finally {
                                        setIsRetrying(false);
                                    }
                                }}
                            >
                                <Loader2 className={cn("w-3 h-3", isRetrying ? "animate-spin" : "")} />
                                {tt("重试", "Retry")}
                            </Button>
                        )}
                        <span className={cn(
                            "text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider h-fit whitespace-nowrap",
                            statusColor
                        )}>
                            {bench.download_status || tt("等待中", "Pending")}
                        </span>
                    </div>
                </div>
                
                {/* Type & Tags */}
                <div className="flex flex-wrap gap-1 mb-3">
                    {evalType && (
                        <span className="text-[10px] bg-blue-50 text-blue-600 px-1.5 py-0.5 rounded border border-blue-100 font-medium">
                            {typeof evalType === 'object' ? JSON.stringify(evalType) : String(evalType)}
                        </span>
                    )}
                    {tags.slice(0, 3).map((tag: any, i: number) => {
                         const tagStr = typeof tag === 'object' ? JSON.stringify(tag) : String(tag);
                         return (
                            <span key={i} className="text-[10px] bg-slate-50 text-slate-500 px-1.5 py-0.5 rounded border border-slate-100 flex items-center gap-1">
                                <Tag className="w-2 h-2" /> {tagStr}
                            </span>
                         );
                    })}
                </div>
                <div className="mb-3 flex items-center gap-2 text-[10px]">
                    <span className={cn(
                        "px-2 py-0.5 rounded-full border",
                        structureReady ? "bg-emerald-50 text-emerald-700 border-emerald-100" :
                        structureFailed ? "bg-red-50 text-red-700 border-red-100" :
                        "bg-amber-50 text-amber-700 border-amber-100"
                    )}>
                        {structureReady ? tt("结构已解析", "Structure Ready") : structureFailed ? tt("结构解析失败", "Structure Failed") : tt("结构解析中", "Parsing Structure")}
                    </span>
                    <span className={cn(
                        "px-2 py-0.5 rounded-full border",
                        downloadReady ? "bg-emerald-50 text-emerald-700 border-emerald-100" :
                        downloadFailed ? "bg-red-50 text-red-700 border-red-100" :
                        "bg-slate-50 text-slate-600 border-slate-100"
                    )}>
                        {downloadReady ? tt("下载完成", "Download Ready") : downloadFailed ? tt("下载失败", "Download Failed") : tt("等待下载", "Waiting Download")}
                    </span>
                </div>

                <div className="text-[10px] text-slate-500 font-mono flex-1 overflow-hidden bg-slate-50/50 p-3 rounded-lg border border-slate-50 group-hover:border-slate-100 transition-colors relative">
                    {hasData ? (
                        <div className="space-y-3 h-full overflow-y-auto pb-4 scrollbar-hide">
                            <div className="text-slate-500 mb-2 font-sans leading-relaxed text-[10px] whitespace-pre-wrap line-clamp-4">
                                {description}
                            </div>

                            {/* Key Mapping Preview */}
                            {keyMapping && isObject(keyMapping) && (
                                <div>
                                    <div className="text-[9px] uppercase font-bold text-slate-400 mb-1">Key Mapping</div>
                                    <div className="grid grid-cols-1 gap-1 pl-2 border-l-2 border-amber-200">
                                        {Object.entries(keyMapping).slice(0, 3).map(([k, v]) => (
                                            <div key={k} className="flex gap-1">
                                                <span className="text-slate-600">{k}:</span>
                                                <span className="text-amber-700 font-bold truncate" title={String(v)}>{String(v)}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Download Info */}
                            {(downloadPath || sampleCount) && (
                                <div className="pt-2 border-t border-slate-200/50 flex justify-between items-center text-[9px] text-slate-400 font-mono">
                                    {sampleCount !== undefined && <span className="flex items-center gap-1"><Database className="w-2 h-2" /> {String(sampleCount)}</span>}
                                    {downloadPath && <span className="truncate max-w-[100px] flex items-center gap-1" title={downloadPath}><SaveIcon className="w-2 h-2" /> ...{downloadPath.slice(-12)}</span>}
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="h-full flex flex-col items-center justify-center text-slate-400 gap-2 min-h-[100px]">
                            {isLoading ? (
                                <>
                                    <Loader2 className="w-4 h-4 animate-spin text-amber-500" />
                                    <span className="text-xs italic">{tt("解析中...", "Analyzing...")}</span>
                                </>
                            ) : (
                                <span className="text-xs italic">{tt("等待解析...", "Waiting for analysis...")}</span>
                            )}
                        </div>
                    )}
                    
                    {/* Hover Hint */}
                    {hasData && (
                        <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-white shadow-sm border border-slate-200 rounded px-2 py-1 text-[10px] text-blue-600 font-bold pointer-events-none flex items-center gap-1">
                            <Maximize2 className="w-3 h-3" /> Configure
                        </div>
                    )}
                </div>
            </div>

            <AnimatePresence>
                {isDetailsOpen && (
                    <Modal
                        isOpen={isDetailsOpen}
                        onClose={() => setIsDetailsOpen(false)}
                        title={
                            <>
                                <Database className="w-6 h-6 text-blue-600" />
                                {bench.bench_name}
                                {evalType && (
                                    <span className="text-sm font-normal text-slate-400 ml-2 bg-slate-100 px-2 py-0.5 rounded-full">
                                        {typeof evalType === 'object' ? JSON.stringify(evalType) : String(evalType)}
                                    </span>
                                )}
                            </>
                        }
                        description={tt("查看并修改该数据集配置。", "Review and modify the benchmark configuration.")}
                        footer={
                            <>
                                <Button variant="ghost" onClick={() => setIsDetailsOpen(false)}>{tt("取消", "Cancel")}</Button>
                                {bench.download_status === "failed" && onRetryDownload && (
                                    <Button
                                        variant="outline"
                                        disabled={isRetrying}
                                        onClick={async () => {
                                            setIsRetrying(true);
                                            try {
                                                await onRetryDownload({ bench_name: bench.bench_name, config: selectedSubset, split: selectedSplit });
                                            } finally {
                                                setIsRetrying(false);
                                            }
                                        }}
                                        className="gap-2"
                                    >
                                        <Loader2 className={cn("w-4 h-4", isRetrying ? "animate-spin" : "")} /> {tt("重试下载", "Retry Download")}
                                    </Button>
                                )}
                                <Button onClick={handleSave} className="bg-blue-600 hover:bg-blue-700 text-white gap-2">
                                    <SaveIcon className="w-4 h-4" /> {tt("保存配置", "Save Configuration")}
                                </Button>
                            </>
                        }
                    >
                        <div className="space-y-8">
                            {bench.download_status === "failed" && (
                                <div className="bg-red-50/50 p-4 rounded-xl border border-red-100">
                                    <div className="text-xs font-bold text-red-700 mb-1">{tt("下载失败", "Download Failed")}</div>
                                    {downloadError && <div className="text-xs text-red-600 font-mono whitespace-pre-wrap">{downloadError}</div>}
                                </div>
                            )}

                            {previewData.length > 0 && (
                                <div className="bg-slate-50 p-5 rounded-xl border border-slate-100">
                                    <h4 className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-blue-500"/> {tt("数据预览", "Dataset Preview")}
                                    </h4>
                                    <div className="overflow-x-auto rounded-lg border border-slate-200">
                                        <table className="w-full text-xs text-left">
                                            <thead className="text-[10px] text-slate-500 uppercase bg-slate-100 font-bold">
                                                <tr>
                                                    {Object.keys(previewData[0] || {}).map(h => <th key={h} className="px-3 py-2 whitespace-nowrap">{h}</th>)}
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-slate-100">
                                                {previewData.slice(0, 5).map((row: any, idx: number) => (
                                                    <tr
                                                        key={idx}
                                                        className={cn("bg-white hover:bg-slate-50 cursor-pointer", selectedPreviewRow === idx ? "bg-blue-50/40" : "")}
                                                        onClick={() => setSelectedPreviewRow(idx)}
                                                    >
                                                        {isObject(row) ? Object.values(row).map((val: any, vi) => (
                                                            <td key={vi} className="px-3 py-2 font-mono text-slate-600 max-w-[220px] truncate border-r border-slate-50 last:border-r-0" title={String(val)}>
                                                                {collapseMiddle(val, 120, 55, 45)}
                                                            </td>
                                                        )) : <td className="px-3 py-2 text-slate-400 italic">{tt("无效数据行", "Invalid Row Data")}</td>}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                        <div className="bg-slate-50 px-3 py-1 text-[10px] text-slate-400 text-center border-t border-slate-200">
                                            {tt(`显示前 ${Math.min(previewData.length, 5)} 行，点击行查看完整内容`, `Showing first ${Math.min(previewData.length, 5)} rows, click row for full view`)}
                                        </div>
                                    </div>
                                    {isObject(previewData[selectedPreviewRow]) && (
                                        <div className="mt-3 p-3 rounded-lg border border-slate-200 bg-white">
                                            <div className="text-[11px] font-bold text-slate-500 mb-2">{tt(`第 ${selectedPreviewRow + 1} 行详情`, `Row ${selectedPreviewRow + 1} Details`)}</div>
                                            <div className="space-y-2 max-h-[260px] overflow-y-auto">
                                                {Object.entries(previewData[selectedPreviewRow]).map(([k, v]) => (
                                                    <div key={k} className="grid grid-cols-12 gap-2 text-xs">
                                                        <div className="col-span-4 text-slate-500 font-mono break-all">{k}</div>
                                                        <div className="col-span-8 text-slate-700 font-mono whitespace-pre-wrap break-all">{collapseMiddle(v, 1800, 1000, 600)}</div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            <div className="bg-indigo-50/50 p-5 rounded-xl border border-indigo-100">
                                <div className="flex items-center justify-between mb-3">
                                    <h4 className="text-sm font-bold text-indigo-800 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-indigo-500"/> {tt("评测类型", "Evaluation Type")}
                                    </h4>
                                    <span className="text-[10px] text-indigo-500 uppercase font-bold tracking-wider">{tt("必填", "Required")}</span>
                                </div>
                                <select
                                    value={selectedEvalType}
                                    onChange={(e) => {
                                        const next = e.target.value;
                                        setSelectedEvalType(next);
                                        const required = EVAL_TYPE_REQUIRED_KEYS[next] || [];
                                        setEditKeyMap(prev => {
                                            const out = { ...prev };
                                            required.forEach(k => {
                                                if (!(k in out)) out[k] = "";
                                            });
                                            return out;
                                        });
                                    }}
                                    className="w-full h-9 rounded-lg border border-indigo-200 bg-white px-3 text-sm font-bold text-slate-900 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all"
                                >
                                    <option value="">{tt("请选择评测类型", "Select evaluation type")}</option>
                                    {EVAL_TYPE_OPTIONS.map(opt => (
                                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                                    ))}
                                </select>
                                <div className="mt-2 text-xs text-indigo-700">
                                    {selectedEvalType
                                        ? (EVAL_TYPE_OPTIONS.find(x => x.value === selectedEvalType)?.desc || "")
                                        : tt("用于决定数据字段如何映射并驱动 DataFlow 评测。", "Defines field mapping and drives DataFlow evaluation.")}
                                </div>
                                {selectedEvalType && (
                                    <div className="mt-3 rounded-lg border border-indigo-100 bg-white p-3 text-xs text-slate-700 space-y-1">
                                        <div><b>{tt("适用场景", "Use case")}:</b> {EVAL_TYPE_GUIDE[selectedEvalType]?.scene}</div>
                                        <div><b>{tt("建议字段", "Suggested fields")}:</b> {EVAL_TYPE_GUIDE[selectedEvalType]?.example}</div>
                                        <div><b>{tt("必填映射", "Required mapping")}:</b> {(EVAL_TYPE_REQUIRED_KEYS[selectedEvalType] || []).join(", ")}</div>
                                    </div>
                                )}
                            </div>
                            
                            {/* 1. Key Mapping Section (Editable) */}
                                <div className="bg-amber-50/50 p-5 rounded-xl border border-amber-100">
                                    <div className="flex justify-between items-center mb-4">
                                        <h4 className="text-sm font-bold text-amber-800 flex items-center gap-2">
                                            <span className="w-2 h-2 rounded-full bg-amber-500"/> Key Mapping
                                        </h4>
                                        {(selectedEvalType || evalType) && (
                                            <div className="text-xs text-amber-700 bg-amber-100 px-2 py-1 rounded border border-amber-200">
                                                {tt("类型", "Type")}: <b>{evalTypeLabel || tt("未选择", "Not selected")}</b>
                                            </div>
                                        )}
                                    </div>
                                    <datalist id={keyOptionsListId}>
                                        {availableKeys.map((ak: any) => (
                                            <option key={String(ak)} value={String(ak)} />
                                        ))}
                                    </datalist>
                                    <div className="grid grid-cols-2 gap-x-8 gap-y-4 text-sm">
                                        {keyMappingEntries.map((k) => (
                                            <div key={k} className="flex items-center gap-3">
                                                <span className="text-slate-500 font-mono w-1/3 text-right text-xs truncate" title={k}>{k}</span>
                                                <Input 
                                                    list={keyOptionsListId}
                                                    value={String((editKeyMap as any)[k] ?? "")}
                                                    onChange={(e) => setEditKeyMap({ ...editKeyMap, [k]: e.target.value })}
                                                    placeholder={tt("支持嵌套路径，如 mc1_targets.choices", "Nested path supported, e.g. mc1_targets.choices")}
                                                    className="h-8 bg-white border-amber-200 focus-visible:ring-amber-500 font-mono text-xs font-bold text-slate-800"
                                                />
                                            </div>
                                        ))}
                                    </div>
                                </div>

                            {/* Local Cache Path */}
                            {downloadPath && (
                                <div className="bg-slate-50 p-5 rounded-xl border border-slate-100">
                                    <h4 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-slate-500"/> {tt("数据集本地存储位置", "Dataset Local Storage Path")}
                                    </h4>
                                    <div className="p-3 rounded-lg border border-slate-200 bg-white text-xs font-mono text-slate-600 break-all">
                                        {downloadPath}
                                    </div>
                                </div>
                            )}


                            {/* 2. Config Selector Section */}
                            {structureSubsets.length > 0 && (
                                <div className="bg-slate-50 p-5 rounded-xl border border-slate-100">
                                    <h4 className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-slate-500"/> {tt("数据集配置", "Dataset Configuration")}
                                    </h4>
                                    
                                    <div className="space-y-4">
                                        {/* Subsets */}
                                        <div>
                                            <label className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 block">{tt("子集 / 配置", "Subset / Config")}</label>
                                            <div className="flex flex-wrap gap-2">
                                                {structureSubsets.map((s: any) => (
                                                    <button
                                                        key={s.subset}
                                                        onClick={() => {
                                                            setSelectedSubset(s.subset);
                                                            // Auto-select first split if available
                                                            if (s.splits && s.splits.length > 0) setSelectedSplit(s.splits[0]);
                                                        }}
                                                        className={cn(
                                                            "px-3 py-1.5 rounded-lg text-xs font-medium border transition-all",
                                                            selectedSubset === s.subset 
                                                                ? "bg-slate-800 text-white border-slate-800 shadow-md" 
                                                                : "bg-white text-slate-600 border-slate-200 hover:border-slate-300"
                                                        )}
                                                    >
                                                        {toLabel(s.subset)}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Splits (Based on selected subset) */}
                                        {selectedSubset && (
                                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 block">Split</label>
                                                <div className="flex flex-wrap gap-2">
                                                    {structureSubsets.find((s: any) => s.subset === selectedSubset)?.splits?.map((split: any) => (
                                                        <button
                                                            key={toLabel(split)}
                                                            onClick={() => setSelectedSplit(toLabel(split))}
                                                            className={cn(
                                                                "px-3 py-1.5 rounded-lg text-xs font-medium border transition-all",
                                                                selectedSplit === toLabel(split)
                                                                    ? "bg-blue-600 text-white border-blue-600 shadow-md" 
                                                                    : "bg-white text-slate-600 border-slate-200 hover:border-slate-300"
                                                            )}
                                                        >
                                                            {toLabel(split)}
                                                        </button>
                                                    ))}
                                                </div>
                                            </motion.div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* 3. Result Preview / Agent Reason */}
                            <div className="bg-blue-50/50 p-5 rounded-xl border border-blue-100">
                                <div className="flex justify-between items-center mb-3">
                                    <h4 className="text-sm font-bold text-blue-800 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-blue-500"/> {tt("最终配置", "Final Configuration")}
                                    </h4>
                                    <span className="text-[10px] text-blue-400 uppercase font-bold tracking-wider">
                                        {tt("推荐配置", "Agent Recommended Configuration")}
                                    </span>
                                </div>
                                <div className="bg-white p-3 rounded-lg border border-blue-100 flex gap-4 items-center">
                                    <div className="flex-1">
                                        <div className="text-[10px] text-slate-400 uppercase">Config</div>
                                        <div className="font-mono font-bold text-blue-700">{selectedSubset || tt("未选择", "Not selected")}</div>
                                    </div>
                                    <div className="w-px h-8 bg-blue-100" />
                                    <div className="flex-1">
                                        <div className="text-[10px] text-slate-400 uppercase">Split</div>
                                        <div className="font-mono font-bold text-blue-700">{selectedSplit || tt("未选择", "Not selected")}</div>
                                    </div>
                                </div>
                                {downloadConfig?.reason && (
                                    <div className="mt-2 text-xs text-blue-600/80 italic pl-1">
                                        "{downloadConfig.reason}"
                                    </div>
                                )}
                            </div>
                            
                            {/* Raw Meta Fallback */}
                            <div className="border-t border-slate-100 pt-4">
                                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">{tt("原始元数据", "Raw Metadata")}</h4>
                                <pre className="text-[10px] text-slate-400 max-h-32 overflow-y-auto bg-slate-50 p-2 rounded border border-slate-100">
                                    {JSON.stringify(bench.meta, null, 2)}
                                </pre>
                            </div>
                        </div>
                    </Modal>
                )}
            </AnimatePresence>
        </>
    );
};

// --- Chat Panel Component ---
interface ChatMessage {
    id: string;
    role: "user" | "ai" | "system";
    content: string | React.ReactNode;
    timestamp: number;
}

interface ChatPanelProps {
    messages: ChatMessage[];
    status: string;
    onSendMessage: (msg: string) => void;
    onConfirm: () => void;
    onStop?: () => void;
    isWaitingForInput: boolean;
    activeNodeId?: string | null;
    isCollapsed: boolean;
    onToggleCollapse: () => void;
    lang: Lang;
    interruptToken?: string | null;
    currentNode?: string | null;
    evalProgress?: Array<{
        bench_name?: string;
        stage?: string;
        generated?: number;
        total?: number;
        percent?: number;
        model?: string;
        model_name?: string;
        current?: number;
        status?: string;
    }> | null;
}

const EMOJIS = ["✨", "🤖", "🚀", "💡", "🔮", "✅", "🎯"];

// 工作流节点相关辅助函数
const NODE_ORDER = [
    "QueryUnderstandNode",
    "BenchSearchNode",
    "HumanReviewNode",
    "DatasetStructureNode",
    "BenchConfigRecommendNode",
    "DownloadNode",
    "DatasetKeysNode",
    "BenchTaskInferNode",
    "PreEvalReviewNode",
    "DataFlowEvalNode",
    "MetricRecommendNode",
    "ScoreCalcNode",
    "ReportGenNode"
];

const getNodeDescription = (nodeName: string | null | undefined, lang: string = "zh"): string => {
    const descriptions: Record<string, { zh: string; en: string }> = {
        "QueryUnderstandNode": { zh: "正在理解您的评测需求...", en: "Understanding your evaluation needs..." },
        "BenchSearchNode": { zh: "正在搜索相关基准测试...", en: "Searching for relevant benchmarks..." },
        "HumanReviewNode": { zh: "等待您的确认...", en: "Waiting for your confirmation..." },
        "DatasetStructureNode": { zh: "正在分析数据集结构...", en: "Analyzing dataset structure..." },
        "BenchConfigRecommendNode": { zh: "正在推荐配置...", en: "Recommending configuration..." },
        "DownloadNode": { zh: "正在下载数据集...", en: "Downloading dataset..." },
        "DatasetKeysNode": { zh: "正在识别数据字段...", en: "Identifying data fields..." },
        "BenchTaskInferNode": { zh: "正在推断评测任务...", en: "Inferring evaluation tasks..." },
        "PreEvalReviewNode": { zh: "等待评测确认...", en: "Waiting for evaluation confirmation..." },
        "DataFlowEvalNode": { zh: "正在执行评测...", en: "Running evaluation..." },
        "MetricRecommendNode": { zh: "正在推荐指标...", en: "Recommending metrics..." },
        "ScoreCalcNode": { zh: "正在计算分数...", en: "Calculating scores..." },
        "ReportGenNode": { zh: "正在生成报告...", en: "Generating report..." },
    };
    const desc = descriptions[nodeName || ""];
    return desc ? (lang === "zh" ? desc.zh : desc.en) : (lang === "zh" ? "处理中..." : "Processing...");
};

const getNodeStage = (nodeName: string | null | undefined, lang: string = "zh"): string => {
    const stages: Record<string, { zh: string; en: string }> = {
        "QueryUnderstandNode": { zh: "阶段 1/7: 需求理解", en: "Phase 1/7: Understanding" },
        "BenchSearchNode": { zh: "阶段 1/7: 基准搜索", en: "Phase 1/7: Benchmark Search" },
        "HumanReviewNode": { zh: "阶段 1/7: 人工审核", en: "Phase 1/7: Human Review" },
        "DatasetStructureNode": { zh: "阶段 2/7: 数据集分析", en: "Phase 2/7: Dataset Analysis" },
        "BenchConfigRecommendNode": { zh: "阶段 2/7: 配置推荐", en: "Phase 2/7: Config Recommendation" },
        "DownloadNode": { zh: "阶段 2/7: 数据下载", en: "Phase 2/7: Data Download" },
        "DatasetKeysNode": { zh: "阶段 3/7: 字段识别", en: "Phase 3/7: Field Identification" },
        "BenchTaskInferNode": { zh: "阶段 3/7: 任务推断", en: "Phase 3/7: Task Inference" },
        "PreEvalReviewNode": { zh: "阶段 4/7: 评测确认", en: "Phase 4/7: Evaluation Review" },
        "DataFlowEvalNode": { zh: "阶段 4/7: 评测执行", en: "Phase 4/7: Evaluation Execution" },
        "MetricRecommendNode": { zh: "阶段 5/7: 指标推荐", en: "Phase 5/7: Metric Recommendation" },
        "ScoreCalcNode": { zh: "阶段 6/7: 分数计算", en: "Phase 6/7: Score Calculation" },
        "ReportGenNode": { zh: "阶段 7/7: 报告生成", en: "Phase 7/7: Report Generation" },
    };
    const stage = stages[nodeName || ""];
    return stage ? (lang === "zh" ? stage.zh : stage.en) : (lang === "zh" ? "处理中" : "Processing");
};

// 从进度数组中获取主要进度信息
// multi-model 场景：SSE 会推送 N 条带 model_name 的 progress。
// 我们聚合所有 entries：terminal(failed/cancelled)=100%、running=its percent、unknown=0。
// 取平均 percent 作为整体进度，避免任何一个 model 卡在 0% 让前端永远显示 0。
const getPrimaryProgress = (evalProgress?: Array<{
    bench_name?: string;
    stage?: string;
    generated?: number;
    total?: number;
    percent?: number;
    model_name?: string;
    model?: string;
    current?: number;
    status?: string;
}> | null): {
    percent?: number;
    bench_name?: string;
    stage?: string;
    generated?: number;
    total?: number;
    model?: string;
    current?: number;
    status?: string;
} | null => {
    if (!evalProgress || !Array.isArray(evalProgress) || evalProgress.length === 0) {
        return null;
    }

    // 第一步：如果只有一个 entry 或没有 model_name，沿用旧的"取第一个 percent"逻辑。
    const withModel = evalProgress.filter(p => (p?.model_name || p?.model));
    if (withModel.length <= 1) {
        const withPercent = evalProgress.find(p => p.percent !== undefined);
        if (withPercent) return withPercent;
        return evalProgress[0];
    }

    // Multi-model：聚合 percent。terminal status 计 100%，否则取 percent（fallback 0）。
    let sum = 0;
    let count = 0;
    let lastBench: string | undefined;
    let lastStage: string | undefined;
    for (const p of withModel) {
        const status = p?.status;
        const pct = typeof p?.percent === "number" ? p.percent : 0;
        const eff = status === "success" || status === "failed" || status === "cancelled" || p?.stage === "done"
            ? 100
            : pct;
        sum += eff;
        count += 1;
        if (p?.bench_name) lastBench = p.bench_name;
        if (p?.stage) lastStage = p.stage;
    }
    const avg = count > 0 ? sum / count : 0;
    return {
        percent: avg,
        bench_name: lastBench,
        stage: lastStage,
    };
};

const getOverallProgress = (nodeName: string | null | undefined, evalProgress?: Array<{
    bench_name?: string;
    stage?: string;
    generated?: number;
    total?: number;
    percent?: number;
    model?: string;
    current?: number;
}> | null): number => {
    if (!nodeName) return 0;

    // 从进度数组中获取主要进度信息
    const primaryProgress = getPrimaryProgress(evalProgress);

    // 如果有评测进度，优先使用
    if (primaryProgress?.percent !== undefined && nodeName === "DataFlowEvalNode") {
        // 评测节点占总进度的 40%，前面节点占 40%，后面占 20%
        const baseProgress = 40;
        return baseProgress + (primaryProgress.percent * 0.4);
    }

    const idx = NODE_ORDER.indexOf(nodeName);
    if (idx === -1) return 10;

    // 根据节点位置计算总进度
    const progressPerNode = 100 / NODE_ORDER.length;
    return Math.min(95, (idx + 1) * progressPerNode);
};

const isNodeCompleted = (currentNode: string | null | undefined, targetNode: string): boolean => {
    if (!currentNode) return false;
    const currentIdx = NODE_ORDER.indexOf(currentNode);
    const targetIdx = NODE_ORDER.indexOf(targetNode);
    return currentIdx > targetIdx;
};

export const ChatPanel = ({ messages, status, onSendMessage, onConfirm, onStop, isWaitingForInput, activeNodeId, isCollapsed, onToggleCollapse, lang, interruptToken, currentNode, evalProgress }: ChatPanelProps) => {
    const [input, setInput] = React.useState("");
    const [dismissedInterrupts, setDismissedInterrupts] = React.useState<string[]>([]);
    const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
    const confirmText = activeNodeId?.includes("PreEvalReviewNode")
        ? tt("请先在执行阶段检查目标模型与评测参数，确认无误后点击批准开始评测。", "Please review the target model and evaluation parameters in the Execution Phase, then approve to start evaluation.")
        : tt("我已准备好基准配置，请检查流程块中的高亮参数。", "I've prepared the benchmark configuration. Please review the highlighted parameters in the workflow blocks.");

    const handleConfirm = () => {
        if (interruptToken) {
            setDismissedInterrupts(prev => prev.includes(interruptToken) ? prev : [...prev, interruptToken]);
        }
        onConfirm();
    };

    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSend = () => {
        if (!input.trim()) return;
        onSendMessage(input);
        setInput("");
    };

    const getEmojiForMessage = (messageId: string) => {
        let hash = 0;
        for (let i = 0; i < messageId.length; i++) {
            hash = (hash * 31 + messageId.charCodeAt(i)) >>> 0;
        }
        return EMOJIS[hash % EMOJIS.length];
    };

    return (
        <motion.div 
            animate={{ width: isCollapsed ? 60 : 400 }}
            className="h-full flex flex-col bg-white/60 backdrop-blur-xl border-l border-white/40 shadow-[-10px_0_30px_-10px_rgba(0,0,0,0.1)] relative overflow-hidden transition-all duration-300"
        >
            {/* Collapse Toggle */}
            <Button 
                variant="ghost" 
                size="icon" 
                onClick={onToggleCollapse}
                className="absolute top-4 right-4 z-50 h-6 w-6 text-slate-400 hover:text-slate-600"
            >
                {isCollapsed ? <ChevronDown className="w-4 h-4 rotate-90" /> : <ChevronDown className="w-4 h-4 -rotate-90" />}
            </Button>

            {isCollapsed ? (
                <div className="flex flex-col items-center pt-20 gap-4">
                    <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-white shadow-lg">
                        <Bot className="w-5 h-5" />
                    </div>
                    <div className={cn("w-2 h-2 rounded-full", status === "running" ? "bg-green-500 animate-pulse" : "bg-slate-300")} />
                </div>
            ) : (
                <>
                    {/* Header */}
                    <div className="p-4 border-b border-white/40 bg-white/30 flex items-center justify-between pr-12">
                        <div className="flex items-center gap-2 text-slate-800 font-bold">
                            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-white shadow-lg shadow-violet-500/20">
                                <Bot className="w-5 h-5" />
                            </div>
                            <span>{tt("Z-Eval 助手", "Z-Eval Assistant")}</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className={cn("w-2 h-2 rounded-full", status === "running" ? "bg-green-500 animate-pulse" : "bg-slate-300")} />
                            <span className="text-xs text-slate-500 uppercase font-medium">{status}</span>
                        </div>
                    </div>

                    {/* Messages */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-6" ref={scrollRef}>
                        {messages.map((msg) => (
                            <motion.div 
                                key={msg.id}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className={cn(
                                    "flex flex-col max-w-[90%]",
                                    msg.role === "user" ? "self-end items-end" : "self-start items-start"
                                )}
                            >
                                <div className="flex items-center gap-2 mb-1 px-1">
                                    {msg.role === "ai" && <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{tt("AI 助手", "AI Assistant")}</span>}
                                    {msg.role === "user" && <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{tt("你", "You")}</span>}
                                </div>
                                
                                <div className={cn(
                                    "px-5 py-3.5 text-sm shadow-sm leading-relaxed",
                                    msg.role === "user" 
                                        ? "bg-gradient-to-br from-blue-600 to-indigo-600 text-white rounded-2xl rounded-tr-sm shadow-blue-500/20" 
                                        : "bg-white border border-white/60 text-slate-700 rounded-2xl rounded-tl-sm shadow-sm"
                                )}>
                                    {msg.role === "ai" && <span className="mr-2">{getEmojiForMessage(msg.id)}</span>}
                                    {msg.content}
                                </div>
                                <span className="text-[10px] text-slate-300 mt-1 px-1">
                                    {new Date(msg.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                                </span>
                            </motion.div>
                        ))}

                        {/* Confirmation Card */}
                        {/* Plan C: Also show approve button when activeNodeId is an interrupt node (handles race condition in RAG mode) */}
                        {status === "interrupted" && interruptToken && !dismissedInterrupts.includes(interruptToken) && (
                            <motion.div 
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="self-start w-full pr-8"
                            >
                                <div className="bg-gradient-to-br from-white to-amber-50/50 border border-amber-100 rounded-2xl p-5 shadow-lg shadow-amber-500/5 ring-1 ring-amber-100/50 relative overflow-hidden">
                                    <div className="absolute top-0 right-0 w-16 h-16 bg-amber-500/10 rounded-full blur-2xl -mr-8 -mt-8" />
                                    
                                    <div className="flex items-center gap-2 mb-3 text-amber-600 font-bold text-sm">
                                        <AlertCircle className="w-4 h-4" />
                                        {tt("需要确认", "Review Required")}
                                    </div>
                                    <p className="text-sm text-slate-600 mb-5 leading-relaxed">
                                        {confirmText}
                                    </p>
                                    <div className="flex gap-3">
                                        <Button size="sm" onClick={handleConfirm} className="flex-1 bg-amber-500 hover:bg-amber-600 text-white shadow-lg shadow-amber-500/20 border-0 rounded-xl h-9">
                                            <Check className="w-4 h-4 mr-1.5" /> {tt("批准", "Approve")}
                                        </Button>
                                        <Button size="sm" variant="outline" onClick={onStop} className="h-9 rounded-xl">
                                            {tt("停止", "Stop")}
                                        </Button>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                        
                        {status === "running" && (
                             <motion.div
                                 initial={{ opacity: 0, y: 10 }}
                                 animate={{ opacity: 1, y: 0 }}
                                 className="self-start w-full pr-8"
                             >
                                 <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-100 rounded-2xl p-4 shadow-sm">
                                     <div className="flex items-center gap-3 mb-3">
                                         <div className="w-8 h-8 rounded-lg bg-blue-500 flex items-center justify-center">
                                             <Loader2 className="w-4 h-4 text-white animate-spin" />
                                         </div>
                                         <div>
                                             <div className="font-semibold text-slate-800 text-sm">{tt("工作流执行中", "Workflow Running")}</div>
                                             <div className="text-xs text-slate-500">{getNodeDescription(currentNode)}</div>
                                         </div>
                                     </div>

                                     {/* 进度条和阶段 */}
                                     <div className="space-y-2">
                                         <div className="flex items-center justify-between text-xs">
                                             <span className="text-slate-600">{getNodeStage(currentNode)}</span>
                                             {(() => {
                                                 const primaryProgress = getPrimaryProgress(evalProgress);
                                                 return primaryProgress && primaryProgress.percent !== undefined && (
                                                     <span className="text-blue-600 font-medium">
                                                         {`${Math.round(primaryProgress.percent)}%`}
                                                     </span>
                                                 );
                                             })()}
                                         </div>

                                         {/* 主进度条 */}
                                         <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                                             <div
                                                 className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full transition-all duration-300"
                                                 style={{ width: `${getOverallProgress(currentNode, evalProgress)}%` }}
                                             />
                                         </div>

                                         {/* 评测进度详情 */}
                                         {(() => {
                                             const primaryProgress = getPrimaryProgress(evalProgress);
                                             if (!primaryProgress) return null;

                                             return (
                                                 <div className="mt-3 p-2 bg-white/50 rounded-lg text-xs space-y-1">
                                                     {primaryProgress.bench_name && (
                                                         <div className="flex items-center justify-between">
                                                             <span className="text-slate-500">{tt("当前评测集", "Current Bench")}</span>
                                                             <span className="font-medium text-slate-700">{primaryProgress.bench_name}</span>
                                                         </div>
                                                     )}
                                                     {primaryProgress.stage && (
                                                         <div className="flex items-center justify-between">
                                                             <span className="text-slate-500">{tt("阶段", "Stage")}</span>
                                                             <span className="font-medium text-slate-700">{primaryProgress.stage}</span>
                                                         </div>
                                                     )}
                                                     {primaryProgress.model && (
                                                         <div className="flex items-center justify-between">
                                                             <span className="text-slate-500">{tt("模型", "Model")}</span>
                                                             <span className="font-medium text-slate-700">{primaryProgress.model}</span>
                                                         </div>
                                                     )}
                                                     {(primaryProgress.generated !== undefined || primaryProgress.current !== undefined) && (primaryProgress.total !== undefined) && (
                                                         <div className="flex items-center justify-between">
                                                             <span className="text-slate-500">{tt("进度", "Progress")}</span>
                                                             <span className="font-medium text-slate-700">
                                                                 {(primaryProgress.generated !== undefined ? primaryProgress.generated : primaryProgress.current || 0)} / {primaryProgress.total}
                                                             </span>
                                                         </div>
                                                     )}
                                                 </div>
                                             );
                                         })()}

                                         {/* 节点步骤指示器 */}
                                         <div className="flex items-center gap-1 mt-3">
                                             {["QueryUnderstandNode", "BenchSearchNode", "DownloadNode", "DataFlowEvalNode", "ScoreCalcNode"].map((node, idx) => (
                                                 <div key={node} className="flex items-center">
                                                     <div className={cn(
                                                         "w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold transition-all",
                                                         isNodeCompleted(currentNode, node)
                                                             ? "bg-emerald-500 text-white"
                                                             : node === currentNode
                                                                 ? "bg-blue-500 text-white ring-2 ring-blue-300"
                                                                 : "bg-slate-200 text-slate-400"
                                                     )}>
                                                         {isNodeCompleted(currentNode, node) ? "✓" : idx + 1}
                                                     </div>
                                                     {idx < 4 && (
                                                         <div className={cn(
                                                             "w-4 h-0.5 transition-all",
                                                             isNodeCompleted(currentNode, ["QueryUnderstandNode", "BenchSearchNode", "DownloadNode", "DataFlowEvalNode"][idx])
                                                                 ? "bg-emerald-500"
                                                                 : "bg-slate-200"
                                                         )} />
                                                     )}
                                                 </div>
                                             ))}
                                         </div>
                                     </div>
                                 </div>
                             </motion.div>
                        )}
                    </div>

                    {/* Input Area */}
                    <div className="p-4 bg-white/40 border-t border-white/20 backdrop-blur-md">
                        <div className="relative group">
                            <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-violet-500 rounded-2xl opacity-20 group-hover:opacity-40 transition duration-500 blur"></div>
                            <div className="relative flex items-center bg-white rounded-xl shadow-sm border border-slate-100 overflow-hidden">
                                <Input 
                                    value={input}
                                    onChange={e => setInput(e.target.value)}
                                    onKeyDown={e => e.key === "Enter" && handleSend()}
                                    placeholder={tt("输入消息...", "Type a message...")}
                                    disabled={isWaitingForInput}
                                    className="border-0 bg-transparent focus-visible:ring-0 text-slate-800 placeholder:text-slate-400 h-12 text-sm px-4 shadow-none"
                                />
                                <Button 
                                    size="icon" 
                                    variant="ghost" 
                                    className="mr-1 h-9 w-9 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                                    onClick={handleSend}
                                    disabled={!input.trim() || isWaitingForInput}
                                >
                                    <Send className="w-4 h-4" />
                                </Button>
                            </div>
                        </div>
                        <div className="text-[10px] text-center text-slate-400 mt-2">
                            {tt("按 Enter 发送", "Press Enter to send")}
                        </div>
                    </div>
                </>
            )}
        </motion.div>
    );
};

// --- Workflow Block Component ---
interface WorkflowBlockProps {
    title: string;
    icon: React.ElementType;
    nodes: { id: string; label: string }[];
    activeNodeId: string | null;
    status: "pending" | "running" | "completed" | "interrupted" | "idle";
    colorTheme: "violet" | "amber" | "emerald";
    lang: Lang;
    children: React.ReactNode;
}

export const WorkflowBlock = ({ title, icon: Icon, nodes, activeNodeId, status, colorTheme, lang, children }: WorkflowBlockProps) => {
    // Theme configurations
    const themes = {
        violet: {
            bg: "bg-violet-50",
            border: "border-violet-100",
            text: "text-violet-700",
            iconBg: "bg-violet-100",
            iconText: "text-violet-600",
            activeBorder: "border-violet-300",
            shadow: "shadow-violet-500/10",
            gradient: "from-violet-50 to-white",
            nodeActive: "bg-violet-500 border-violet-500 text-white"
        },
        amber: {
            bg: "bg-amber-50",
            border: "border-amber-100",
            text: "text-amber-700",
            iconBg: "bg-amber-100",
            iconText: "text-amber-600",
            activeBorder: "border-amber-300",
            shadow: "shadow-amber-500/10",
            gradient: "from-amber-50 to-white",
            nodeActive: "bg-amber-500 border-amber-500 text-white"
        },
        emerald: {
            bg: "bg-emerald-50",
            border: "border-emerald-100",
            text: "text-emerald-700",
            iconBg: "bg-emerald-100",
            iconText: "text-emerald-600",
            activeBorder: "border-emerald-300",
            shadow: "shadow-emerald-500/10",
            gradient: "from-emerald-50 to-white",
            nodeActive: "bg-emerald-500 border-emerald-500 text-white"
        }
    };
    
    const theme = themes[colorTheme];
    const isBlockActive = nodes.some(n => n.id === activeNodeId) || status === "completed" || status === "interrupted" || status === "running"; 
    
    // Auto-expand if active or completed, but also if it contains meaningful content (hacky check: if status is idle, collapse)
    // Actually just use isBlockActive logic which is fine.

    return (
        <motion.div 
            layout
            className={cn(
                "rounded-[2rem] border transition-all duration-500 overflow-hidden flex flex-col relative group",
                isBlockActive 
                    ? `bg-white border-transparent shadow-xl ${theme.shadow} ring-1 ring-slate-100` 
                    : "bg-white/60 border-slate-100 shadow-sm opacity-80 hover:opacity-100"
            )}
        >
            {/* Header / Node Visualization */}
            <div className={cn("p-6 border-b border-slate-50 bg-gradient-to-b", theme.gradient)}>
                <div className="flex justify-between items-start mb-8">
                    <div className="flex items-center gap-4">
                        <div className={cn(
                            "w-12 h-12 rounded-2xl flex items-center justify-center shadow-sm",
                            theme.iconBg, theme.iconText
                        )}>
                            <Icon className="w-6 h-6" />
                        </div>
                        <div>
                            <h3 className={cn("font-bold text-xl tracking-tight", isBlockActive ? "text-slate-900" : "text-slate-500")}>{title}</h3>
                            <div className="flex items-center gap-2 mt-1">
                                <span className={cn("w-2 h-2 rounded-full", isBlockActive ? theme.nodeActive.split(' ')[0] : "bg-slate-300")} />
                                <span className="text-xs text-slate-400 font-bold uppercase tracking-wider">{lang === "zh" ? "阶段" : "Phase"}</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Nodes Stepper */}
                <div className="flex items-center justify-between relative px-4 pb-2">
                    {/* Connecting Line */}
                    <div className="absolute top-1/2 left-6 right-6 h-0.5 bg-slate-100 -z-0" />
                    
                    {nodes.map((node, index) => {
                        const isNodeActive = activeNodeId === node.id;
                        // Determine if node is "passed" (completed)
                        // A simple heuristic: if activeNodeId is later in the list than this node
                        const activeIndex = nodes.findIndex(n => n.id === activeNodeId);
                        const isPassed = activeIndex > -1 && index < activeIndex;
                        const isCompletedStatus = status === "completed";

                        return (
                            <div key={node.id} className="relative z-10 flex flex-col items-center gap-3 group/node cursor-pointer">
                                <motion.div 
                                    className={cn(
                                        "w-10 h-10 rounded-full flex items-center justify-center border-4 transition-all duration-300 bg-white shadow-sm relative",
                                        isNodeActive 
                                            ? `${theme.nodeActive} shadow-lg scale-110` 
                                            : (isPassed || isCompletedStatus) 
                                                ? `${theme.text.replace('text-', 'border-')} ${theme.text}` 
                                                : "border-slate-100 text-slate-300"
                                    )}
                                    whileHover={{ scale: 1.1 }}
                                >
                                    {isNodeActive && (
                                        <span className="absolute inset-0 rounded-full animate-ping opacity-75 bg-current" />
                                    )}
                                    {(isPassed || isCompletedStatus) && !isNodeActive && (
                                         <div className={cn("w-2 h-2 rounded-full relative z-10", theme.nodeActive.split(' ')[0])} />
                                    )}
                                    <div className={cn("w-2 h-2 rounded-full relative z-10", isNodeActive ? "bg-white" : ((isPassed || isCompletedStatus) ? "hidden" : "bg-slate-300"))} />
                                </motion.div>
                                <div className="flex flex-col items-center">
                                    <span className={cn(
                                        "text-[10px] font-bold uppercase tracking-wider transition-colors px-2 py-1 rounded-md",
                                        isNodeActive ? `${theme.iconBg} ${theme.text}` : ((isPassed || isCompletedStatus) ? theme.text : "text-slate-400")
                                    )}>
                                        {node.label}
                                    </span>
                                    {isNodeActive && (
                                        <motion.span 
                                            initial={{ opacity: 0, y: -5 }} 
                                            animate={{ opacity: 1, y: 0 }}
                                            className="text-[9px] text-slate-400 font-medium mt-0.5 flex items-center gap-1"
                                        >
                                            <Loader2 className="w-2 h-2 animate-spin" /> {lang === "zh" ? "运行中" : "Running"}
                                        </motion.span>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Content / Config Area */}
            <div className="p-8 bg-white flex-1 relative">
                {isBlockActive && (
                    <div className={cn("absolute top-0 left-0 w-1 h-full", theme.iconBg)} />
                )}
                {children}
            </div>
        </motion.div>
    );
};



// --- Judge Entry Component (LLM-as-Judge trigger) ---
// Note: JudgeEntry removed — entry point moved to Tab 3 "裁判评分"


// --- Summary Bench Card Component ---
const SummaryBenchCard = ({
    bench,
    lang,
    workflowMeta,
    onRetryModel,
}: {
    bench: any;
    lang: Lang;
    workflowMeta?: WorkflowMetaSnapshot | null;
    onRetryModel?: (benchName: string, modelName: string) => void;
}) => {
    const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
    const [isResultOpen, setIsResultOpen] = useState(true);
    // const [isSummaryOpen, setIsSummaryOpen] = useState(false);

    // 支持多模型结果（新 schema：eval_results[model] 带 status/error/stats/detail_path）
    const evalResults = bench.meta?.eval_results || {};  // 多模型: {model_name: {status, stats, detail_path, error, ...}}
    const singleResult = bench.meta?.eval_result;  // 单模型兼容
    const hasMultipleResults = Object.keys(evalResults).length > 0;
    const hasSingleResult = singleResult && Object.keys(singleResult).length > 0;
    // const hasSummary = !!bench.meta?.metric_summary;

    // 获取所有模型的名字（含失败 / 取消的）
    const modelNames = hasMultipleResults ? Object.keys(evalResults) : (hasSingleResult ? ["Model"] : []);

    // 从 workflow_meta 优先取 status（更精确，eval_results 也有 status 兜底）
    const metaBench = workflowMeta?.benches?.[bench.bench_name];
    const getModelStatus = (name: string): string | undefined => {
        const fromMeta = metaBench?.models?.[name]?.status;
        if (fromMeta) return fromMeta;
        return (evalResults[name] as any)?.status;
    };
    const getModelError = (name: string): string | undefined => {
        const fromMeta = metaBench?.models?.[name]?.error;
        if (typeof fromMeta === "string" && fromMeta) return fromMeta;
        return (evalResults[name] as any)?.error;
    };
    const getModelFailureCount = (name: string): { cf?: number; mf?: number; lastError?: string | null } => {
        const meta = metaBench?.models?.[name];
        const slot = (evalResults[name] as any) || {};
        return {
            cf: meta?.consecutive_failures ?? slot.consecutive_failures,
            mf: meta?.max_failures ?? slot.max_failures,
            lastError: meta?.last_error ?? slot.last_error ?? null,
        };
    };

    // 整体圆点颜色：success / partial / failed / running / pending
    const overallStatus = bench.eval_status;
    const overallDotCls =
        overallStatus === "success"
            ? "bg-emerald-500"
            : overallStatus === "partial"
                ? "bg-amber-500"
                : overallStatus === "failed"
                    ? "bg-red-500"
                    : overallStatus === "running"
                        ? "bg-blue-500 animate-pulse"
                        : "bg-slate-200";

    return (
        <div className={cn(
          "bg-white rounded-xl p-3 border border-slate-200 shadow-sm flex flex-col gap-3 relative group hover:border-blue-400 hover:shadow-md transition-all self-start",
          hasMultipleResults ? "min-w-[400px]" : "min-w-[320px]"
        )}>
            <div className="flex justify-between items-start">
                <span className="font-bold text-sm text-slate-800 line-clamp-1" title={bench.bench_name}>{bench.bench_name}</span>
                <div
                    className={cn("w-2 h-2 rounded-full ring-2 ring-white shrink-0 ml-2", overallDotCls)}
                    title={overallStatus || ""}
                />
            </div>

            <div className="space-y-2">
                {/* Multi-Model Results Section */}
                {hasMultipleResults ? (
                    <div className="border border-slate-100 rounded-lg overflow-hidden bg-slate-50/30">
                        <div
                            className="bg-slate-50 px-3 py-2 flex justify-between items-center cursor-pointer hover:bg-slate-100 transition-colors"
                            onClick={() => setIsResultOpen(!isResultOpen)}
                        >
                            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1">
                                <Tag className="w-3 h-3" /> {tt("多模型对比", "Model Comparison")} ({modelNames.length})
                            </span>
                            {isResultOpen ? <ChevronUp className="w-3 h-3 text-slate-400"/> : <ChevronDown className="w-3 h-3 text-slate-400"/>}
                        </div>
                        {isResultOpen && (
                             <div className="p-3 bg-white text-xs border-t border-slate-100 overflow-x-auto">
                                 <table className="w-full border-collapse">
                                     <thead>
                                         <tr className="border-b border-slate-200">
                                             <th className="text-left py-1 px-2 text-slate-500 font-medium">{tt("指标", "Metric")}</th>
                                             {modelNames.map(name => {
                                                 const st = getModelStatus(name);
                                                 const err = getModelError(name);
                                                 const fc = getModelFailureCount(name);
                                                 const canRetry = (st === "failed" || st === "cancelled") && !!onRetryModel;
                                                 return (
                                                     <th key={name} className="text-right py-1 px-2 text-slate-700 font-semibold align-bottom">
                                                         <div className="flex items-center justify-end gap-1.5 flex-wrap">
                                                             <span className="truncate max-w-[120px]" title={err ? `${name}\n${err}` : name}>{name}</span>
                                                             <ModelStatusBadge status={st} lang={lang} />
                                                             <FailureCounterBadge
                                                                 consecutive={fc.cf}
                                                                 max={fc.mf}
                                                                 lastError={fc.lastError}
                                                                 lang={lang}
                                                             />
                                                             {canRetry && (
                                                                 <button
                                                                     type="button"
                                                                     onClick={() => onRetryModel?.(bench.bench_name, name)}
                                                                     className="w-5 h-5 rounded-full flex items-center justify-center text-blue-500 hover:bg-blue-50 transition-colors"
                                                                     title={tt("重试该模型", "Retry this model")}
                                                                 >
                                                                     <RotateCw className="w-3 h-3" />
                                                                 </button>
                                                             )}
                                                         </div>
                                                     </th>
                                                 );
                                             })}
                                         </tr>
                                     </thead>
                                     <tbody>
                                         {(() => {
                                             // 收集所有指标
                                             const allMetrics = new Set<string>();
                                             Object.values(evalResults).forEach((r: any) => {
                                                 if (r?.stats) Object.keys(r.stats).forEach(k => allMetrics.add(k));
                                             });
                                             return [...allMetrics].map(metric => (
                                                 <tr key={metric} className="border-b border-slate-50 last:border-0">
                                                     <td className="py-1 px-2 text-slate-500 truncate">{metric}</td>
                                                     {modelNames.map(name => {
                                                         const v = (evalResults[name] as any)?.stats?.[metric];
                                                         return (
                                                             <td key={name} className="text-right py-1 px-2 font-mono font-bold text-emerald-600">
                                                                 {v !== undefined ? (typeof v === 'number' ? v.toFixed(4) : String(v)) : '-'}
                                                             </td>
                                                         );
                                                     })}
                                                 </tr>
                                             ));
                                         })()}
                                     </tbody>
                                 </table>
                             </div>
                        )}
                    </div>
                ) : hasSingleResult ? (
                    /* Single Model Result Section */
                    <div className="border border-slate-100 rounded-lg overflow-hidden bg-slate-50/30">
                        <div
                            className="bg-slate-50 px-3 py-2 flex justify-between items-center cursor-pointer hover:bg-slate-100 transition-colors"
                            onClick={() => setIsResultOpen(!isResultOpen)}
                        >
                            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1">
                                <Tag className="w-3 h-3" /> {tt("指标结果", "Metric Result")}
                            </span>
                            {isResultOpen ? <ChevronUp className="w-3 h-3 text-slate-400"/> : <ChevronDown className="w-3 h-3 text-slate-400"/>}
                        </div>
                        {isResultOpen && (
                             <div className="p-3 bg-white text-xs space-y-1 border-t border-slate-100">
                                 {Object.entries(singleResult).map(([k, v]) => (
                                     <div key={k} className="flex justify-between items-center border-b border-slate-50 last:border-0 pb-1 last:pb-0">
                                         <span className="text-slate-500 font-medium truncate pr-2" title={k}>{k}</span>
                                         <span className="font-mono font-bold text-emerald-600">
                                            {typeof v === 'number' ? v.toFixed(4) : String(v)}
                                         </span>
                                     </div>
                                 ))}
                             </div>
                        )}
                    </div>
                ) : (
                    <div className="h-10 flex items-center justify-center text-[10px] text-slate-300 italic border border-dashed border-slate-100 rounded-lg">
                        {tt("暂无结果", "No results yet")}
                    </div>
                )}

                {/* Metric Summary Section */}
                {/* {hasSummary && (
                    <div className="border border-slate-100 rounded-lg overflow-hidden bg-slate-50/30">
                        <div
                            className="bg-slate-50 px-3 py-2 flex justify-between items-center cursor-pointer hover:bg-slate-100 transition-colors"
                            onClick={() => setIsSummaryOpen(!isSummaryOpen)}
                        >
                            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1">
                                <Bot className="w-3 h-3" /> Metric Summary
                            </span>
                            {isSummaryOpen ? <ChevronUp className="w-3 h-3 text-slate-400"/> : <ChevronDown className="w-3 h-3 text-slate-400"/>}
                        </div>
                        {isSummaryOpen && (
                             <div className="p-3 bg-white border-t border-slate-100 max-h-[300px] overflow-y-auto scrollbar-thin scrollbar-thumb-slate-200">
                                 <SimpleMarkdown content={bench.meta.metric_summary} />
                             </div>
                        )}
                    </div>
                )} */}

                {/* LLM Judge Entry — moved to Tab 3 */}
            </div>
        </div>
    );
};

// --- Judge Report View (Tab 4: renders judge report data natively) ---
const JudgeReportView = ({ report, lang }: { report: any; lang: Lang }) => {
    const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);

    const ranking = report.overall?.ranking || [];
    const radar = report.macro?.radar || {};
    const errorDist = report.diagnostic?.error_distribution || {};
    const summary = report.analyst?.summary || report.llm_summary || "";

    return (
        <div className="space-y-6 animate-in fade-in duration-500 pb-10">
            {/* Header */}
            <div className="bg-gradient-to-r from-slate-900 to-slate-800 rounded-2xl p-6 text-white shadow-xl relative overflow-hidden">
                <div className="absolute top-0 right-0 w-48 h-48 bg-white/5 rounded-full blur-3xl -mr-12 -mt-12 pointer-events-none" />
                <div className="relative z-10">
                    <div className="flex items-center gap-2 mb-1">
                        <ClipboardCheck className="w-5 h-5 text-blue-400" />
                        <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                            {tt("裁判评分分析报告", "Judge Analysis Report")}
                        </span>
                    </div>
                    <h2 className="text-xl font-bold text-white">{report.bench_name || tt("评测报告", "Evaluation Report")}</h2>
                    {report.generated_at && (
                        <p className="text-xs text-slate-400 mt-1">
                            {tt("生成时间", "Generated")}: {report.generated_at}
                        </p>
                    )}
                </div>
            </div>

            {/* Ranking Table */}
            {ranking.length > 0 && (
                <div className="bg-white rounded-xl border border-slate-100 p-5 shadow-sm">
                    <h3 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-2">
                        <Tag className="w-4 h-4 text-blue-500" />
                        {tt("模型排名", "Model Rankings")}
                    </h3>
                    <table className="w-full text-xs">
                        <thead>
                            <tr className="border-b border-slate-200 text-left">
                                <th className="py-2 px-2 text-slate-500 w-10">#</th>
                                <th className="py-2 px-2 text-slate-500">{tt("模型", "Model")}</th>
                                <th className="py-2 px-2 text-right text-slate-500">{tt("总分", "Overall")}</th>
                                <th className="py-2 px-2 text-right text-slate-500">{tt("推理", "Think")}</th>
                                <th className="py-2 px-2 text-right text-slate-500">{tt("回答", "Body")}</th>
                            </tr>
                        </thead>
                        <tbody>
                            {ranking.map((r: any, i: number) => (
                                <tr key={i} className="border-b border-slate-50 hover:bg-slate-50">
                                    <td className="py-2 px-2">
                                        <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${
                                            i === 0 ? 'bg-amber-100 text-amber-700' :
                                            i === 1 ? 'bg-slate-100 text-slate-600' :
                                            i === 2 ? 'bg-orange-50 text-orange-600' :
                                            'text-slate-400'
                                        }`}>{i + 1}</span>
                                    </td>
                                    <td className="py-2 px-2 font-medium text-slate-700">{r.model}</td>
                                    <td className="py-2 px-2 text-right font-mono font-bold text-emerald-600">
                                        {typeof r.overall_score === 'number' ? r.overall_score.toFixed(2) : r.overall_score}
                                    </td>
                                    <td className="py-2 px-2 text-right font-mono text-blue-600">
                                        {typeof r.think_score === 'number' ? r.think_score.toFixed(2) : r.think_score}
                                    </td>
                                    <td className="py-2 px-2 text-right font-mono text-blue-600">
                                        {typeof r.body_score === 'number' ? r.body_score.toFixed(2) : r.body_score}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Radar / Score Comparison */}
            {radar.dimensions && radar.data && Object.keys(radar.data).length > 0 && (
                <div className="bg-white rounded-xl border border-slate-100 p-5 shadow-sm">
                    <h3 className="text-sm font-bold text-slate-700 mb-3">{tt("维度对比", "Dimension Comparison")}</h3>
                    <div className="space-y-3">
                        {radar.dimensions.map((dim: string) => (
                            <div key={dim} className="space-y-1">
                                <div className="text-xs text-slate-500 font-medium">{dim}</div>
                                <div className="flex items-center gap-3">
                                    {Object.entries(radar.data).map(([model, scores]: [string, any]) => {
                                        const val = (scores as any)?.[dim] || 0;
                                        const pct = Math.min((val / 5) * 100, 100);
                                        return (
                                            <div key={model} className="flex-1">
                                                <div className="flex items-center justify-between text-[10px] mb-0.5">
                                                    <span className="text-slate-500 truncate">{model}</span>
                                                    <span className="font-mono text-slate-600">{typeof val === 'number' ? val.toFixed(2) : val}</span>
                                                </div>
                                                <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                                                    <div className="h-full bg-blue-500 rounded-full transition-all" style={{ width: `${pct}%` }} />
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Error Distribution */}
            {Object.keys(errorDist).length > 0 && (
                <div className="bg-white rounded-xl border border-slate-100 p-5 shadow-sm">
                    <h3 className="text-sm font-bold text-slate-700 mb-3">{tt("问题分布", "Issue Distribution")}</h3>
                    <div className="space-y-4">
                        {Object.entries(errorDist).map(([model, issues]: [string, any]) => {
                            const issueEntries = Object.entries(issues || {}).sort(([,a]: any, [,b]: any) => b - a);
                            if (issueEntries.length === 0) return null;
                            return (
                                <div key={model}>
                                    <div className="text-xs font-bold text-slate-600 mb-1.5">{model}</div>
                                    <div className="space-y-1 pl-3 border-l-2 border-slate-100">
                                        {issueEntries.slice(0, 8).map(([issue, count]: [string, any]) => (
                                            <div key={issue} className="flex items-center justify-between text-xs">
                                                <span className="text-slate-500 truncate max-w-[300px]" title={issue}>{issue}</span>
                                                <span className="text-slate-400 font-mono ml-2 shrink-0">{count}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* LLM Analysis */}
            {summary && (
                <div className="bg-white rounded-xl border border-slate-100 p-5 shadow-sm">
                    <h3 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-2">
                        <Bot className="w-4 h-4 text-violet-500" />
                        {tt("AI 分析", "AI Analysis")}
                    </h3>
                    <div className="text-xs text-slate-600 leading-relaxed whitespace-pre-wrap">
                        <SimpleMarkdown content={summary} />
                    </div>
                </div>
            )}
        </div>
    );
};

// --- Summary Panel Component (4 tabs) ---
type ViewMode = "benches" | "output" | "judge" | "report";

interface JudgeTabState {
  judgeBench: any;
  judgeModels: { model_name: string; detail_path: string }[];
  judgeTaskId: string | null;
  judgeStatus: 'idle' | 'running' | 'completed' | 'failed' | null;
  judgeHistory: any[];
  judgeReport: any;
  judgeReportStatus: 'idle' | 'generating' | 'done' | 'error';
}

const initialJudgeTabState: JudgeTabState = {
  judgeBench: null,
  judgeModels: [],
  judgeTaskId: null,
  judgeStatus: null,
  judgeHistory: [],
  judgeReport: null,
  judgeReportStatus: 'idle',
};

export const SummaryPanel = ({
  state, threadId, sidebarWidth, chatWidth, lang, apiBaseUrl,
  judgeTabState, onJudgeAction,
  onViewJudgeResult,
  isFullscreen, onFullscreenChange,
  onPreviewModel, onDownloadModel,
  workflowMeta, sseAlive, onRetryModel,
}: {
    state: WorkflowState | null,
    threadId: string | null,
    sidebarWidth: number,
    chatWidth: number,
    lang: Lang,
    apiBaseUrl: string,
    judgeTabState: JudgeTabState,
    onJudgeAction: (action: string, payload?: any) => void,
    onViewJudgeResult?: () => void,
    isFullscreen?: boolean,
    onFullscreenChange?: (fullscreen: boolean) => void,
    onPreviewModel?: (bench: string, model: string) => void,
    onDownloadModel?: (bench: string, model: string) => void,
    workflowMeta?: WorkflowMetaSnapshot | null,
    sseAlive?: boolean,
    onRetryModel?: (benchName: string, modelName: string) => void,
}) => {
    const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
    const [isOpen, setIsOpen] = React.useState(true);
    const [viewMode, setViewMode] = React.useState<ViewMode>("benches");

    const fullscreen = isFullscreen ?? false;

    if (!state) return null;

    const hasReport = state.reports && state.reports["default"];
    const hasModelOutput = state.benches?.some((b: any) => {
      const er = b.meta?.eval_results || {};
      return Object.keys(er).length > 0 || b.meta?.eval_detail_path;
    });
    const hasJudgeResult = judgeTabState.judgeStatus === 'completed' && judgeTabState.judgeTaskId;

    const isTabDisabled = (tab: ViewMode): boolean => {
      if (tab === "judge") return !hasModelOutput;
      if (tab === "report") return !hasJudgeResult;
      return false;
    };

    const tabClass = (tab: ViewMode, activeColor = "text-slate-600 border-slate-600") =>
      cn(
        "flex items-center gap-2 text-xs font-bold uppercase tracking-wider h-full border-b-2 transition-all cursor-pointer px-2",
        viewMode === tab ? activeColor : "text-slate-400 border-transparent hover:text-slate-500",
        isTabDisabled(tab) && "opacity-40 cursor-not-allowed hover:text-slate-400"
      );

    const handleTabClick = (tab: ViewMode) => {
      if (isTabDisabled(tab)) return;
      setViewMode(tab);
      setIsOpen(true);
    };

    return (
        <motion.div
            initial={{ y: 100 }}
            animate={{
                y: 0,
                left: sidebarWidth + 60,
                right: chatWidth + 0,
                height: !isOpen ? 48 : fullscreen ? "100vh" : "60vh"
            }}
            className="fixed bottom-0 z-40 px-8 pb-0 pointer-events-none transition-all duration-300"
        >
            <div className="max-w-6xl mx-auto pointer-events-auto h-full flex flex-col">
                <div className="bg-white/90 backdrop-blur-xl border border-slate-200 rounded-t-2xl shadow-[0_-10px_40px_-15px_rgba(0,0,0,0.1)] overflow-hidden ring-1 ring-slate-100 flex flex-col h-full transition-all duration-300">
                    {/* Tab Header */}
                    <div className="h-12 bg-slate-50/50 border-b border-slate-100 flex items-center justify-between px-6 shrink-0">
                        <div className="flex items-center gap-5 h-full">
                            {/* Tab 1: 上下文 */}
                            <div className={tabClass("benches")} onClick={() => handleTabClick("benches")}>
                                <Database className="w-4 h-4" />
                                {tt("上下文", "Context")}
                            </div>

                            {/* Tab 2: 模型输出 */}
                            <div className={tabClass("output")} onClick={() => handleTabClick("output")}>
                                <Bot className="w-4 h-4" />
                                {tt("模型输出", "Model Output")}
                            </div>

                            {/* Tab 3: 裁判评分 */}
                            <div className={tabClass("judge", "text-blue-600 border-blue-600")} onClick={() => handleTabClick("judge")}>
                                <ClipboardCheck className="w-4 h-4" />
                                {tt("裁判评分", "LLM Judge")}
                            </div>

                            {/* Tab 4: 分析报告 */}
                            <div className={tabClass("report", "text-violet-600 border-violet-600 bg-violet-50/50")} onClick={() => handleTabClick("report")}>
                                <BarChart3 className="w-4 h-4" />
                                {tt("分析报告", "Analysis Report")}
                            </div>
                        </div>

                        <div className="flex items-center gap-1">
                            {/* 展出全部 / 退出全屏 */}
                            {isOpen && (
                                <div
                                    className="flex items-center gap-1.5 cursor-pointer p-2 hover:bg-slate-100 rounded-lg transition-colors"
                                    onClick={() => onFullscreenChange?.(!fullscreen)}
                                >
                                    {fullscreen
                                        ? <Minimize2 className="w-3.5 h-3.5 text-slate-400" />
                                        : <Maximize2 className="w-3.5 h-3.5 text-slate-400" />
                                    }
                                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                                        {fullscreen ? tt("退出全屏", "Exit Full") : tt("展出全部", "Full View")}
                                    </span>
                                </div>
                            )}
                            {/* 收起 / 展开 */}
                            <div
                                className="flex items-center gap-1.5 cursor-pointer p-2 hover:bg-slate-100 rounded-lg transition-colors"
                                onClick={() => setIsOpen(!isOpen)}
                            >
                                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                                    {isOpen ? tt("收起", "Collapse") : tt("展开", "Expand")}
                                </span>
                                {isOpen ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronUp className="w-4 h-4 text-slate-400" />}
                            </div>
                        </div>
                    </div>

                    {/* Tab Content */}
                    <AnimatePresence mode="wait">
                        {isOpen && (
                            <motion.div
                                key={viewMode}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                transition={{ duration: 0.2 }}
                                className="overflow-hidden flex-1 flex flex-col bg-slate-50/30 min-h-0"
                            >
                                {/* Tab 1: 上下文 */}
                                {viewMode === "benches" && (
                                    <TabErrorBoundary fallbackLabel={tt("上下文渲染出错，请刷新", "Context render error, please refresh")}>
                                        <div className="p-6 overflow-y-auto flex-1 scrollbar-thin scrollbar-thumb-slate-200">
                                            {/* 顶部状态条：SSE 实时 vs 轮询；workflow_meta 顶层 status / cancel_reason */}
                                            {(workflowMeta?.status || sseAlive !== undefined) && (
                                                <div className="flex items-center gap-2 mb-4 text-[10px]">
                                                    {sseAlive ? (
                                                        <span className="bg-emerald-50 text-emerald-700 border border-emerald-100 px-1.5 py-0.5 rounded font-bold uppercase tracking-wider">
                                                            LIVE
                                                        </span>
                                                    ) : (
                                                        <span className="bg-slate-50 text-slate-500 border border-slate-100 px-1.5 py-0.5 rounded font-bold uppercase tracking-wider">
                                                            POLL
                                                        </span>
                                                    )}
                                                    {workflowMeta?.status && (
                                                        <span className="text-slate-500">
                                                            {tt("流程状态", "Workflow")}: <b className="text-slate-700">{workflowMeta.status}</b>
                                                        </span>
                                                    )}
                                                    {workflowMeta?.cancel_reason && (
                                                        <span className="text-red-600" title={workflowMeta.cancel_reason}>
                                                            {tt("取消原因", "Cancelled")}: {workflowMeta.cancel_reason}
                                                        </span>
                                                    )}
                                                </div>
                                            )}
                                            <div className="flex gap-4 pb-4 scrollbar-hide items-start">
                                                {state.benches?.length ? state.benches.map((b: any, i: number) => (
                                                    <SummaryBenchCard
                                                        key={i}
                                                        bench={b}
                                                        lang={lang}
                                                        workflowMeta={workflowMeta}
                                                        onRetryModel={onRetryModel}
                                                    />
                                                )) : (
                                                    <div className="h-24 w-full border-2 border-dashed border-slate-100 rounded-xl flex items-center justify-center text-xs text-slate-400">
                                                        {tt("暂无已选基准", "No benchmarks selected")}
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </TabErrorBoundary>
                                )}

                                {/* Tab 2: 模型输出 */}
                                {viewMode === "output" && (
                                    <TabErrorBoundary fallbackLabel={tt("模型输出渲染出错，请刷新", "Model Output render error, please refresh")}>
                                        <div className="p-6 overflow-y-auto flex-1 scrollbar-thin scrollbar-thumb-slate-200">
                                            {(() => {
                                              // Collect all models across all benches. Include failed / cancelled
                                              // entries too (status badge will distinguish them); only filter when
                                              // there's literally no eval_results dict.
                                              type OutputEntry = {
                                                  bench: string;
                                                  model: string;
                                                  stats: Record<string, any>;
                                                  status?: string;
                                                  error?: string;
                                                  consecutive_failures?: number;
                                                  max_failures?: number;
                                                  last_error?: string | null;
                                              };
                                              const entries: OutputEntry[] = [];
                                              state.benches?.forEach((b: any) => {
                                                const er = b.meta?.eval_results || {};
                                                for (const [name, info] of Object.entries(er)) {
                                                  const i = (info as any) || {};
                                                  entries.push({
                                                      bench: b.bench_name,
                                                      model: name,
                                                      stats: i.stats || {},
                                                      status: i.status,
                                                      error: i.error,
                                                      consecutive_failures: i.consecutive_failures,
                                                      max_failures: i.max_failures,
                                                      last_error: i.last_error,
                                                  });
                                                }
                                              });
                                              if (entries.length === 0 && state.reports?.["default"]) {
                                                return (
                                                  <div className="max-w-5xl mx-auto">
                                                    <ReportView report={state.reports["default"]} threadId={threadId} lang={lang} />
                                                  </div>
                                                );
                                              }
                                              if (entries.length === 0) {
                                                return (
                                                  <div className="flex flex-col items-center justify-center py-16 text-slate-400">
                                                    <Bot className="w-12 h-12 mb-3 opacity-30" />
                                                    <p className="text-sm">{tt("暂无模型输出数据", "No model output data yet")}</p>
                                                  </div>
                                                );
                                              }
                                              return (
                                                <div className="max-w-5xl mx-auto space-y-4">
                                                  {entries.map(({ bench, model, stats, status: st, error: err, consecutive_failures: cf, max_failures: mf, last_error: le }) => {
                                                    const isFailed = st === "failed" || st === "cancelled";
                                                    const canRetry = isFailed && !!onRetryModel;
                                                    return (
                                                      <div
                                                          key={`${bench}-${model}`}
                                                          className={cn(
                                                              "bg-white rounded-lg border p-4 shadow-sm",
                                                              isFailed ? "border-red-200" : "border-slate-100",
                                                          )}
                                                      >
                                                        <div className="flex items-center justify-between mb-2">
                                                          <div className="flex items-center gap-2 flex-wrap">
                                                            <span className="text-sm font-bold text-slate-700">{bench}</span>
                                                            <span className="text-[10px] bg-blue-50 text-blue-600 px-1.5 py-0.5 rounded">{model}</span>
                                                            <ModelStatusBadge status={st} lang={lang} />
                                                            <FailureCounterBadge
                                                                consecutive={cf}
                                                                max={mf}
                                                                lastError={le || err}
                                                                lang={lang}
                                                            />
                                                          </div>
                                                          <div className="flex items-center gap-2">
                                                            {!isFailed && (
                                                              <>
                                                                <button
                                                                  type="button"
                                                                  className="text-[10px] bg-blue-500 hover:bg-blue-600 text-white px-2 py-1 rounded-md font-medium transition-colors flex items-center gap-1 cursor-pointer"
                                                                  onClick={() => onPreviewModel?.(bench, model)}
                                                                ><Eye className="w-3 h-3" /> {tt("预览", "Preview")}</button>
                                                                <button
                                                                  type="button"
                                                                  className="text-[10px] bg-emerald-500 hover:bg-emerald-600 text-white px-2 py-1 rounded-md font-medium transition-colors flex items-center gap-1 cursor-pointer"
                                                                  onClick={() => onDownloadModel?.(bench, model)}
                                                                ><Download className="w-3 h-3" /> {tt("下载", "Download")}</button>
                                                              </>
                                                            )}
                                                            {canRetry && (
                                                              <button
                                                                type="button"
                                                                className="text-[10px] bg-amber-500 hover:bg-amber-600 text-white px-2 py-1 rounded-md font-medium transition-colors flex items-center gap-1 cursor-pointer"
                                                                onClick={() => onRetryModel?.(bench, model)}
                                                                title={tt("重试该模型", "Retry this model")}
                                                              ><RotateCw className="w-3 h-3" /> {tt("重试", "Retry")}</button>
                                                            )}
                                                          </div>
                                                        </div>
                                                        {isFailed && err && (
                                                          <div className="text-[11px] text-red-600 font-mono whitespace-pre-wrap break-all bg-red-50/50 border border-red-100 rounded px-2 py-1.5 mb-2">
                                                            {err}
                                                          </div>
                                                        )}
                                                        {Object.keys(stats).length > 0 ? (
                                                          <div className="flex flex-wrap gap-2">
                                                            {Object.entries(stats)
                                                              .filter(([k]) => !["bench_name_or_prefix", "metric", "type"].includes(k))
                                                              .map(([k, v]) => (
                                                                <span key={k} className="text-xs bg-slate-50 px-2 py-1 rounded border border-slate-100">
                                                                  <span className="text-slate-400">{k}:</span>{" "}
                                                                  <span className="font-mono font-bold text-slate-600">{typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(4)) : String(v)}</span>
                                                                </span>
                                                              ))
                                                            }
                                                          </div>
                                                        ) : !isFailed ? (
                                                          <span className="text-xs text-slate-400 italic">{tt("仅生成结果（无评估指标）", "Generation only (no eval metrics)")}</span>
                                                        ) : null}
                                                      </div>
                                                    );
                                                  })}
                                                </div>
                                              );
                                            })()}
                                        </div>
                                    </TabErrorBoundary>
                                )}

                                {/* Tab 3: 裁判评分 */}
                                {viewMode === "judge" && (
                                  <TabErrorBoundary fallbackLabel={tt("裁判评分渲染出错，请刷新", "Judge tab render error, please refresh")}>
                                    <JudgeTabContent
                                      state={state}
                                      lang={lang}
                                      apiBaseUrl={apiBaseUrl}
                                      threadId={threadId}
                                      judgeTabState={judgeTabState}
                                      onJudgeAction={onJudgeAction}
                                      onViewJudgeResult={onViewJudgeResult}
                                    />
                                  </TabErrorBoundary>
                                )}

                                {/* Tab 4: 分析报告 */}
                                {viewMode === "report" && (
                                  <TabErrorBoundary fallbackLabel={tt("分析报告渲染出错，请刷新", "Report tab render error, please refresh")}>
                                    <div className="p-6 overflow-y-auto flex-1 scrollbar-thin scrollbar-thumb-slate-200">
                                      <div className="max-w-5xl mx-auto">
                                        {judgeTabState.judgeReportStatus === 'generating' ? (
                                          <div className="flex flex-col items-center justify-center py-16 text-slate-400">
                                            <Loader2 className="w-8 h-8 animate-spin text-blue-500 mb-3" />
                                            <p className="text-sm">{tt("正在生成分析报告...", "Generating analysis report...")}</p>
                                            <p className="text-xs text-slate-300 mt-1">{tt("LLM 正在分析评分数据", "LLM is analyzing judge results")}</p>
                                          </div>
                                        ) : judgeTabState.judgeReport ? (
                                          <JudgeReportView report={judgeTabState.judgeReport} lang={lang} />
                                        ) : (
                                          <div className="flex flex-col items-center justify-center py-16 text-slate-400">
                                            <BarChart3 className="w-12 h-12 mb-3 opacity-30" />
                                            <p className="text-sm">{tt("请先完成裁判评分，然后生成分析报告", "Complete judging first, then generate the analysis report")}</p>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  </TabErrorBoundary>
                                )}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </motion.div>
    );
};


// --- Judge Tab Content (Tab 3) ---
const JudgeTabContent = ({
  state, lang, apiBaseUrl, threadId, judgeTabState, onJudgeAction, onViewJudgeResult,
}: {
  state: WorkflowState,
  lang: Lang,
  apiBaseUrl: string,
  threadId: string | null,
  judgeTabState: JudgeTabState,
  onJudgeAction: (action: string, payload?: any) => void,
  onViewJudgeResult?: () => void,
}) => {
    const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);

    // Collect all models with results across all benches
    const allModels: { model_name: string; detail_path: string }[] = [];
    const benchesWithResults: any[] = [];
    state.benches?.forEach((b: any) => {
      const er = b.meta?.eval_results || {};
      if (Object.keys(er).length > 0 || b.meta?.eval_detail_path) {
        benchesWithResults.push(b);
        if (Object.keys(er).length > 0) {
          for (const [name, info] of Object.entries(er)) {
            const path = (info as any)?.detail_path;
            if (path && !allModels.find(m => m.model_name === name)) {
              allModels.push({ model_name: name, detail_path: path });
            }
          }
        } else if (b.meta?.eval_detail_path) {
          if (!allModels.find(m => m.model_name === "Model")) {
            allModels.push({ model_name: "Model", detail_path: b.meta.eval_detail_path });
          }
        }
      }
    });

    const { judgeStatus, judgeTaskId, judgeHistory, judgeReportStatus } = judgeTabState;

    // Phase: idle -> running -> completed
    const isRunning = judgeStatus === 'running';
    const isCompleted = judgeStatus === 'completed';

    return (
      <div className="p-6 overflow-y-auto flex-1 space-y-4">
        {/* Phase 1: Config + Start */}
        {!isRunning && !isCompleted && (
          <>
            {allModels.length > 0 ? (
              <>
                <div className="flex items-center gap-2 mb-2">
                  <ClipboardCheck className="w-4 h-4 text-blue-500" />
                  <span className="text-sm font-bold text-slate-700">{tt("裁判评分配置", "Judge Scoring Config")}</span>
                </div>
                <JudgeConfigPanel
                  apiBaseUrl={apiBaseUrl}
                  models={allModels}
                  lang={lang}
                  onStart={(config: any) => onJudgeAction("start", config)}
                />
              </>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-slate-400">
                <ClipboardCheck className="w-10 h-10 mb-2 opacity-30" />
                <p className="text-sm">{tt("请先运行模型评测，获取模型输出数据", "Run model evaluation first to get output data")}</p>
              </div>
            )}

            {/* History */}
            {judgeHistory.length > 0 && (
              <div className="border-t border-slate-100 pt-4 mt-4">
                <div className="flex items-center gap-2 mb-2">
                  <History className="w-4 h-4 text-slate-400" />
                  <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">{tt("历史评分记录", "Judge History")}</span>
                </div>
                <div className="space-y-1.5">
                  {judgeHistory.map((h: any) => (
                    <div key={h.task_id}
                      className="flex items-center justify-between px-3 py-2 rounded-lg border border-slate-100 bg-white hover:bg-slate-50 cursor-pointer transition-colors"
                      onClick={() => onJudgeAction("loadHistory", h)}
                    >
                      <div className="flex items-center gap-3">
                        <span className={`w-2 h-2 rounded-full ${h.status === 'completed' ? 'bg-emerald-500' : h.status === 'failed' ? 'bg-red-400' : 'bg-amber-400'}`} />
                        <span className="text-xs text-slate-600">{h.bench_name}</span>
                        <span className="text-[10px] text-slate-400">{h.started_at?.slice(0, 16)}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-slate-400">{(h.model_names || []).join(', ')}</span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded ${h.status === 'completed' ? 'bg-emerald-50 text-emerald-600' : 'bg-red-50 text-red-500'}`}>
                          {h.status}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* Phase 2: Running */}
        {isRunning && judgeTaskId && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
              <span className="text-sm font-bold text-slate-700">{tt("评分进行中...", "Judging in progress...")}</span>
            </div>
            <JudgeProgressPanel
              apiBaseUrl={apiBaseUrl}
              taskId={judgeTaskId}
              onComplete={() => onJudgeAction("complete")}
              lang={lang}
            />
          </div>
        )}

        {/* Phase 3: Completed */}
        {isCompleted && judgeTaskId && (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Check className="w-4 h-4 text-emerald-500" />
              <span className="text-sm font-bold text-slate-700">{tt("评分完成", "Judging Complete")}</span>
            </div>

            {/* Summary */}
            <div className="bg-emerald-50/50 border border-emerald-200 rounded-lg px-4 py-3">
              <div className="flex items-center justify-between">
                <span className="text-xs text-emerald-700">{tt("评分已完成，可查看详细结果或生成分析报告", "Judging complete. View details or generate analysis report.")}</span>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={() => onViewJudgeResult?.()}
                className="px-4 py-2 rounded-lg text-xs font-semibold text-blue-600 bg-blue-50 border border-blue-200 hover:bg-blue-100 transition-colors"
              >
                {tt("查看详细结果", "View Details")}
              </button>
              <button
                onClick={() => onJudgeAction("generateReport")}
                disabled={judgeReportStatus === 'generating'}
                className="px-4 py-2 rounded-lg text-xs font-semibold text-violet-600 bg-violet-50 border border-violet-200 hover:bg-violet-100 transition-colors disabled:opacity-50"
              >
                {judgeReportStatus === 'generating'
                  ? tt("生成中...", "Generating...")
                  : judgeReportStatus === 'done'
                    ? tt("重新生成报告", "Regenerate Report")
                    : tt("生成分析报告", "Generate Report")}
              </button>
              <button
                onClick={() => onJudgeAction("reset")}
                className="px-4 py-2 rounded-lg text-xs text-slate-500 hover:bg-slate-100 border border-slate-200 transition-colors"
              >
                {tt("重新评分", "Re-judge")}
              </button>
            </div>
          </div>
        )}
      </div>
    );
};

// --- Gallery Modal ---
// 适配 bench_gallery.json 数据结构
export const GalleryModal = ({ isOpen, onClose, onSelect, apiBaseUrl, lang }: { isOpen: boolean, onClose: () => void, onSelect: (bench: any) => void, apiBaseUrl: string, lang: Lang }) => {
    const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
    const [benches, setBenches] = useState<any[]>([]);
    const [search, setSearch] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        if (isOpen) {
            setIsLoading(true);
            fetch(`${apiBaseUrl}/api/benches/gallery`)
                .then(res => res.json())
                .then(data => setBenches(data))
                .catch(err => console.error(err))
                .finally(() => setIsLoading(false));
        }
    }, [isOpen, apiBaseUrl]);

    // 从 bench_gallery.json 获取描述和标签
    const getDescription = (b: any): string => {
        // 优先从 meta.description 获取，中文模式下优先使用 description_zh
        if (lang === 'zh' && b.meta?.description_zh) return b.meta.description_zh;
        if (b.meta?.description) return b.meta.description;
        // 兼容旧格式
        if (typeof b.description === 'string') return b.description;
        return '';
    };

    const getTags = (b: any): string[] => {
        // 优先从 meta.tags 获取
        if (Array.isArray(b.meta?.tags)) return b.meta.tags;
        // 兼容旧格式 task_type
        if (Array.isArray(b.task_type)) return b.task_type;
        return [];
    };

    const getDisplayName = (b: any): string => {
        // 优先从 meta.aliases 获取显示名称
        if (Array.isArray(b.meta?.aliases) && b.meta.aliases.length > 1) {
            return b.meta.aliases[1]; // 第二个通常是大写版本
        }
        return b.bench_name || '';
    };

    const getCategory = (b: any): string => {
        return b.meta?.category || '';
    };

    const isUploaded = (b: any) =>
        b?.meta?.source === "local_upload" ||
        (b?.meta?.artifact_paths && b.meta.artifact_paths.length > 0) ||
        b?.bench_source_url?.startsWith("local://");

    const filtered = Array.isArray(benches) ? benches.filter(b => {
        if (!b) return false;
        const searchLower = search.toLowerCase();
        const benchName = (b.bench_name || "").toLowerCase();
        const description = getDescription(b).toLowerCase();
        const tags = getTags(b).join(" ").toLowerCase();
        const category = getCategory(b).toLowerCase();
        return benchName.includes(searchLower) ||
               description.includes(searchLower) ||
               tags.includes(searchLower) ||
               category.includes(searchLower);
    }).sort((a, b) => {
        // 手动上传的排在前面
        const aUploaded = isUploaded(a);
        const bUploaded = isUploaded(b);
        if (aUploaded && !bUploaded) return -1;
        if (!aUploaded && bUploaded) return 1;
        return 0;
    }) : [];

    return (
        <Modal
            isOpen={isOpen}
            onClose={onClose}
            title={tt("基准库", "Benchmark Gallery")}
            description={tt("选择一个基准并加入当前评测。", "Select a benchmark to add to your evaluation.")}
        >
            <div className="space-y-4">
                <Input
                    placeholder={tt("搜索基准...", "Search benchmarks...")}
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                    className="bg-slate-50"
                />

                {isLoading ? (
                    <div className="flex justify-center py-8"><Loader2 className="animate-spin text-slate-300" /></div>
                ) : (
                    <div className="grid grid-cols-1 gap-2 max-h-[400px] overflow-y-auto pr-2">
                        {filtered.map(b => {
                            const tags = getTags(b);
                            const category = getCategory(b);
                            const uploaded = isUploaded(b);
                            return (
                                <div key={b.bench_name} className={cn("flex items-center justify-between p-3 rounded-lg border transition-all cursor-pointer group",
                                    uploaded ? "border-emerald-200 bg-emerald-50/30 hover:border-emerald-300" : "border-slate-100 hover:border-blue-200 hover:bg-blue-50/30"
                                )}
                                    onClick={() => onSelect(b)}
                                >
                                    <div className="flex-1 min-w-0">
                                        <div className="font-bold text-slate-700 text-sm flex items-center gap-2 flex-wrap">
                                            {uploaded && <Upload className="w-3.5 h-3.5 text-emerald-600" />}
                                            {getDisplayName(b)}
                                            {uploaded && (
                                                <span className="text-[10px] bg-emerald-100 text-emerald-700 px-1.5 rounded">
                                                    {tt("本地上传", "Uploaded")}
                                                </span>
                                            )}
                                            {category && (
                                                <span className="text-[10px] bg-blue-100 text-blue-600 px-1.5 rounded">
                                                    {category}
                                                </span>
                                            )}
                                            {tags.slice(0, 2).map((t: string, idx: number) => (
                                                <span key={idx} className="text-[10px] bg-slate-100 text-slate-500 px-1.5 rounded">
                                                    {t}
                                                </span>
                                            ))}
                                            {tags.length > 2 && (
                                                <span className="text-[10px] bg-slate-100 text-slate-400 px-1.5 rounded">
                                                    +{tags.length - 2}
                                                </span>
                                            )}
                                        </div>
                                        <div className="text-xs text-slate-400 line-clamp-1 mt-0.5">
                                            {getDescription(b)}
                                        </div>
                                    </div>
                                    <Button size="sm" variant="ghost" className="opacity-0 group-hover:opacity-100 shrink-0 ml-2">
                                        {tt("添加", "Add")}
                                    </Button>
                                </div>
                            );
                        })}
                        {filtered.length === 0 && <div className="text-center text-slate-400 py-4">{tt("未找到匹配基准", "No benchmarks found")}</div>}
                    </div>
                )}
            </div>
        </Modal>
    );
};
