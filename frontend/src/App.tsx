import { useEffect, useMemo, useState } from 'react';
import {
  AlertCircle,
  BarChart3,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  Database,
  Download,
  FileQuestion,
  FileText,
  Gauge,
  LayoutDashboard,
  Loader2,
  LogOut,
  MoreVertical,
  MessageSquareText,
  Plus,
  RefreshCw,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  TrendingDown,
} from 'lucide-react';
import {
  askQuestion,
  clearChatHistory,
  clearToken,
  crawlWebCorpus,
  deleteChatHistoryItem,
  deleteDocument,
  deleteWebCorpusItem,
  detectWebDuplicates,
  getCitationUrl,
  getDocuments,
  getAdminInsights,
  getAdminMetrics,
  getChatHistory,
  getChatSession,
  getRetrievalSettings,
  getStatus,
  getToken,
  getDocumentUrl,
  ingest,
  login,
  me,
  resolveAdminInsight,
  sendDemoFeedback,
  sendFeedback,
  sendHumanAudit,
  setToken,
  trackCitationClick,
  updateRetrievalSettings,
  uploadDocuments,
} from './api';
import type {
  AdminInsights,
  AdminMetrics,
  ChatHistoryItem,
  ChatHistorySession,
  ChatResponse,
  DemoFeedbackPayload,
  DocumentItem,
  HumanAuditPayload,
  InsightIssueType,
  RetrievalSettings,
  StatusResponse,
  Strategy,
  User,
  WebDuplicateReport,
} from './types';

interface ConversationItem {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  response?: ChatResponse;
}

const SUGGESTED_QUESTIONS = [
  'What risks are described in the MiCAR risk disclosure?',
  'How does Kraken safeguard client crypto-assets?',
  'What does the privacy notice say about how my personal data is used?',
];

type AuditScoreKey =
  | 'groundedness_score'
  | 'citation_score'
  | 'relevance_score'
  | 'completeness_score'
  | 'clarity_score';

const AUDIT_DIMENSIONS: Array<{ key: AuditScoreKey; label: string; help: string }> = [
  {
    key: 'groundedness_score',
    label: 'Groundedness',
    help: 'Is the answer supported by the cited evidence?',
  },
  {
    key: 'citation_score',
    label: 'Citation',
    help: 'Do the citations point to the right document, chunk, or page?',
  },
  {
    key: 'relevance_score',
    label: 'Relevance',
    help: 'Does the answer address the question asked?',
  },
  {
    key: 'completeness_score',
    label: 'Completeness',
    help: 'Does it include the important details needed?',
  },
  {
    key: 'clarity_score',
    label: 'Clarity',
    help: 'Is the answer easy to understand?',
  },
];

const AUDIT_ISSUES = [
  { value: 'wrong_document', label: 'Wrong document retrieved' },
  { value: 'citation_mismatch', label: 'Citation does not support answer' },
  { value: 'missing_detail', label: 'Missing important detail' },
  { value: 'too_broad', label: 'Answer too broad' },
  { value: 'too_vague', label: 'Answer too vague' },
  { value: 'unsupported_claim', label: 'Unsupported claim' },
  { value: 'unnecessary_refusal', label: 'Refusal was unnecessary' },
  { value: 'should_refuse', label: 'Should have refused' },
  { value: 'good_answer', label: 'Good answer, no major issue' },
  { value: 'unable_to_judge', label: 'Unable to judge' },
];

function scorePercent(score?: number) {
  if (score === undefined) return 'N/A';
  return `${Math.round(Math.max(0, Math.min(1, score)) * 100)}%`;
}

function boostLabel(score?: number | null) {
  if (score === undefined || score === null) return 'N/A';
  return `+${Math.round(Math.max(0, score) * 100)} pts`;
}

function answerQualityScore(response?: ChatResponse) {
  const raw = response?.judge?.overall_score;
  if (typeof raw === 'number') return raw;
  if (typeof raw === 'string') {
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function answerQualityLevel(score?: number) {
  if (score === undefined) return 'unknown';
  if (score >= 4) return 'strong';
  if (score >= 3) return 'okay';
  return 'low';
}

function answerQualityLabel(score?: number) {
  if (score === undefined) return 'not judged';
  if (score >= 4) return 'good answer';
  if (score >= 3) return 'needs review';
  return 'weak answer';
}

function answerQualityDisplay(score?: number) {
  if (score === undefined) return 'N/A';
  return `${Math.round(score)}/5`;
}

function metricScoreDisplay(score?: number | null) {
  if (score === undefined || score === null) return 'N/A';
  return `${score.toFixed(1)}/5`;
}

function judgeString(response: ChatResponse, key: string) {
  const value = response.judge?.[key];
  return typeof value === 'string' ? value : undefined;
}

function judgeNumber(response: ChatResponse, key: string) {
  const value = response.judge?.[key];
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function judgeList(response: ChatResponse, key: string) {
  const value = response.judge?.[key];
  if (Array.isArray(value)) return value.map((item) => String(item)).filter(Boolean);
  if (typeof value === 'string' && value.trim()) return [value];
  return [];
}

function judgeClaimChecks(response: ChatResponse) {
  const value = response.judge?.claim_checks;
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is Record<string, unknown> => typeof item === 'object' && item !== null)
    .map((item) => ({
      claim: String(item.claim ?? 'Claim not provided'),
      citations: Array.isArray(item.citations_used)
        ? item.citations_used.map((citation) => String(citation)).join(', ')
        : String(item.citations_used ?? 'No citation listed'),
      support: String(item.support_status ?? 'not assessed'),
      issue: String(item.issue ?? ''),
    }));
}

function strategyLabel(value?: Strategy | string | null) {
  if (value === 'fixed_size') return 'Fixed-size';
  if (value === 'structure_aware') return 'Structure-aware';
  if (value === 'hybrid') return 'Hybrid ensemble';
  return 'Unknown';
}

function AnswerQualityPill({ response }: { response?: ChatResponse }) {
  const score = answerQualityScore(response);
  const level = answerQualityLevel(score);
  return (
    <span className={`confidencePill ${level}`}>
      Answer quality {answerQualityDisplay(score)}
    </span>
  );
}

function InsightStatusCard({
  label,
  value,
  detail,
  icon,
}: {
  label: string;
  value: string | number;
  detail: string;
  icon: React.ReactNode;
}) {
  return (
    <span className="insightStatusCard">
      <i>{icon}</i>
      <em>{label}</em>
      <strong>{value}</strong>
      <small>{detail}</small>
    </span>
  );
}

function InsightQuestionRows({
  items,
  empty,
  mode,
  resolvingKey,
  onResolve,
}: {
  items: AdminInsights['unanswered_or_refused_questions'] | AdminInsights['low_human_audit_score_questions'];
  empty: string;
  mode: 'llm' | 'human' | 'refused';
  resolvingKey?: string | null;
  onResolve: (issueType: InsightIssueType, itemKey: string, resolved: boolean) => Promise<void>;
}) {
  if (items.length === 0) {
    return <p className="insightEmpty">{empty}</p>;
  }

  return (
    <div className="insightQuestionList">
      {items.map((item) => {
        const isHuman = 'human_average_score' in item;
        const score = isHuman ? item.human_average_score : item.llm_score;
        const verdict = isHuman ? item.overall_verdict : item.llm_verdict;
        const rowKey = `${item.issue_type}:${item.item_key}`;
        return (
          <article className={`insightQuestionRow ${item.resolved ? 'resolved' : ''}`} key={`${mode}-${item.chat_log_id}`}>
            <div>
              <strong>{item.question}</strong>
              <p>{item.answer_preview}</p>
              {item.resolved && (
                <p className="insightNote">
                  Resolved by {item.resolved_by_email ?? 'admin'}
                  {item.resolved_at ? ` on ${formatRecentDate(item.resolved_at)}` : ''}
                </p>
              )}
              {isHuman && item.comment && <p className="insightNote">{item.comment}</p>}
            </div>
            <aside>
              <span>{item.category}</span>
              <span>{strategyLabel(item.strategy)}</span>
              <span>{score === undefined || score === null ? 'No score' : `${Number(score).toFixed(1)}/5`}</span>
              {verdict && <span>{verdict}</span>}
              <small>{formatRecentDate(item.created_at)}</small>
              <button
                className="insightResolveButton"
                disabled={resolvingKey === rowKey}
                onClick={() => onResolve(item.issue_type, item.item_key, !item.resolved)}
              >
                {resolvingKey === rowKey ? 'Saving...' : item.resolved ? 'Unresolve' : 'Resolve'}
              </button>
            </aside>
          </article>
        );
      })}
    </div>
  );
}

function AdminInsightsDashboard({
  insights,
  loading,
  notice,
  onRefresh,
  showResolved,
  onShowResolvedChange,
  resolvingKey,
  onResolve,
}: {
  insights: AdminInsights | null;
  loading: boolean;
  notice?: string;
  onRefresh: () => Promise<void>;
  showResolved: boolean;
  onShowResolvedChange: (value: boolean) => Promise<void>;
  resolvingKey?: string | null;
  onResolve: (issueType: InsightIssueType, itemKey: string, resolved: boolean) => Promise<void>;
}) {
  const maxCategoryCount = Math.max(1, ...(insights?.top_question_categories.map((item) => item.total) ?? [1]));
  const llmInsights = insights?.llm_generated_insights;

  return (
    <section className="insightsDashboard">
      <section className="insightsMain">
        <div className="insightHero">
          <div>
            <span>Admin insights dashboard</span>
            <h1>User demand, refusals, and corpus gaps</h1>
            <p>
              Review question patterns from live testing, identify weak answers, and decide which documents or web
              pages should be added before the next evaluation pass.
            </p>
          </div>
          <button className="secondaryButton compactAction" onClick={() => void onRefresh()} disabled={loading}>
            {loading ? <Loader2 className="spin" size={17} /> : <RefreshCw size={17} />}
            Refresh
          </button>
        </div>

        <div className="insightToolbar">
          <label className="insightToggle">
            <input
              type="checkbox"
              checked={showResolved}
              onChange={(event) => {
                void onShowResolvedChange(event.target.checked);
              }}
            />
            <span>Show resolved flags</span>
          </label>
          {insights?.resolved_counts && (
            <small>
              Resolved: refused {insights.resolved_counts.refused ?? 0}, low LLM{' '}
              {insights.resolved_counts.low_llm ?? 0}, low human {insights.resolved_counts.low_human ?? 0}, gaps{' '}
              {insights.resolved_counts.missing_document ?? 0}
            </small>
          )}
        </div>

        {notice && <p className="notice">{notice}</p>}

        {!insights && !loading && (
          <div className="corpusEmpty">
            <BarChart3 size={24} />
            <strong>No insights loaded yet.</strong>
            <span>Refresh after users have submitted chat questions or human audit reviews.</span>
          </div>
        )}

        {insights && (
          <>
            {llmInsights && (
              <section className="llmInsightCard">
                <div className="insightPanelHeader">
                  <strong>LLM-generated insights</strong>
                  <span>
                    Generated from the dashboard metrics below
                    {llmInsights.model ? ` | ${llmInsights.model}` : ''}
                  </span>
                </div>
                <h2>{llmInsights.headline}</h2>
                <p>{llmInsights.executive_summary}</p>
                <div className="llmInsightColumns">
                  <div>
                    <strong>Key takeaways</strong>
                    <ul>
                      {llmInsights.key_takeaways.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <strong>Suggested actions</strong>
                    <ul>
                      {llmInsights.suggested_actions.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <strong>Watchlist</strong>
                    <ul>
                      {llmInsights.watchlist.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>
                {!llmInsights.available && llmInsights.error && <p className="insightNote">{llmInsights.error}</p>}
              </section>
            )}

            <div className="insightStatusGrid">
              <InsightStatusCard
                label="Questions analysed"
                value={insights.summary.total_questions_analyzed}
                detail="Chat log rows"
                icon={<MessageSquareText size={17} />}
              />
              <InsightStatusCard
                label="Refused"
                value={insights.summary.total_refusals}
                detail="Could not answer from corpus"
                icon={<FileQuestion size={17} />}
              />
              <InsightStatusCard
                label="Low LLM judge"
                value={insights.summary.low_llm_score_questions}
                detail="Overall score 3/5 or below"
                icon={<Gauge size={17} />}
              />
              <InsightStatusCard
                label="Low human audit"
                value={insights.summary.low_human_score_questions}
                detail="Partial, fail, or low score"
                icon={<TrendingDown size={17} />}
              />
              <InsightStatusCard
                label="Improvement topics"
                value={insights.summary.corpus_gap_topics}
                detail="Repeated warning themes"
                icon={<BookOpen size={17} />}
              />
            </div>

            <div className="insightGrid">
              <section className="insightPanel wide">
                <div className="insightPanelHeader">
                  <strong>Top question categories</strong>
                  <span>Bars show relative question volume, scaled against the most common category.</span>
                </div>
                <p className="insightExplainer">
                  The volume bar compares each category against the most common category. The warnings bar compares
                  refusals, low LLM scores, and low human-review counts against that category's question count.
                </p>
                <div className="categoryRankList">
                  {insights.top_question_categories.map((item) => {
                    const warningCount = item.refusals + item.low_llm + item.low_human;
                    const warningRate = item.total > 0 ? Math.min(100, (warningCount / item.total) * 100) : 0;
                    const volumeRate = Math.max(6, (item.total / maxCategoryCount) * 100);
                    return (
                      <article className="categoryRankRow" key={item.category}>
                        <div>
                          <strong>{item.category}</strong>
                          <span>
                            {item.total} question{item.total === 1 ? '' : 's'} | warning signals {warningCount} |{' '}
                            {Math.round(warningRate)}% flagged
                          </span>
                          <span>
                            Refusals {item.refusals} | low LLM {item.low_llm} | low human {item.low_human}
                          </span>
                        </div>
                        <div className="categoryBarStack" aria-hidden="true">
                          <div className="categoryBarLine">
                            <span>Volume</span>
                            <div className="categoryBar volumeBar">
                              <i style={{ width: `${volumeRate}%` }} />
                            </div>
                          </div>
                          <div className="categoryBarLine">
                            <span>Warnings</span>
                            <div className="categoryBar warningBar">
                              <i style={{ width: `${Math.max(warningRate > 0 ? 6 : 0, warningRate)}%` }} />
                            </div>
                          </div>
                        </div>
                        <small>
                          LLM {metricScoreDisplay(item.avg_llm_score)} | Human {metricScoreDisplay(item.avg_human_score)}
                        </small>
                      </article>
                    );
                  })}
                  {insights.top_question_categories.length === 0 && (
                    <p className="insightEmpty">No question categories yet.</p>
                  )}
                </div>
              </section>

              <section className="insightPanel">
                <div className="insightPanelHeader">
                  <strong>Improvement queue</strong>
                  <span>Grouped warning patterns and next review actions</span>
                </div>
                <p className="insightExplainer">
                  A signal is a warning. Examples are: refused answers, no citations, low LLM judge scores, wrong
                  document feedback, missing detail feedback, or citation mismatch feedback. Suggested actions are
                  review starting points, not final judgements.
                </p>
                <div className="gapSignalList">
                  {insights.common_missing_documents.map((item) => {
                    const fallbackRate = item.category_total ? item.count / item.category_total : null;
                    const warningPercent = Math.round(((item.warning_rate ?? fallbackRate) ?? 0) * 100);
                    return (
                      <article className={`gapSignalRow ${item.resolved ? 'resolved' : ''}`} key={item.topic}>
                        <div className="gapSignalTitleRow">
                          <strong>{item.topic}</strong>
                          {item.suggested_action && (
                            <em className={`gapActionPill ${item.action_type ?? 'answer_review'}`}>
                              {item.suggested_action}
                            </em>
                          )}
                        </div>
                        <span>
                          {item.count} signal{item.count === 1 ? '' : 's'}
                          {item.category_total
                            ? ` from ${item.category_total} questions | ${warningPercent}% flagged`
                            : ''}
                        </span>
                        {item.representative_question && <p>{item.representative_question}</p>}
                        <div>
                          {item.signals.map((signal) => (
                            <em key={signal}>{signal}</em>
                          ))}
                        </div>
                        {item.resolved && (
                          <p className="insightNote">
                            Resolved by {item.resolved_by_email ?? 'admin'}
                            {item.resolved_at ? ` on ${formatRecentDate(item.resolved_at)}` : ''}
                          </p>
                        )}
                        <button
                          className="insightResolveButton"
                          disabled={resolvingKey === `${item.issue_type}:${item.item_key}`}
                          onClick={() => onResolve(item.issue_type, item.item_key, !item.resolved)}
                        >
                          {resolvingKey === `${item.issue_type}:${item.item_key}`
                            ? 'Saving...'
                            : item.resolved
                              ? 'Unresolve'
                              : 'Resolve'}
                        </button>
                      </article>
                    );
                  })}
                  {insights.common_missing_documents.length === 0 && (
                    <p className="insightEmpty">No improvement items yet.</p>
                  )}
                </div>
              </section>

              <section className="insightPanel">
                <div className="insightPanelHeader">
                  <strong>Unanswered or refused</strong>
                  <span>Questions the assistant could not answer</span>
                </div>
                <InsightQuestionRows
                  items={insights.unanswered_or_refused_questions}
                  mode="refused"
                  empty="No refused questions found."
                  resolvingKey={resolvingKey}
                  onResolve={onResolve}
                />
              </section>

              <section className="insightPanel">
                <div className="insightPanelHeader">
                  <strong>Low LLM judge score</strong>
                  <span>Potential weak answers needing review</span>
                </div>
                <InsightQuestionRows
                  items={insights.low_llm_judge_score_questions}
                  mode="llm"
                  empty="No low LLM judge score questions found."
                  resolvingKey={resolvingKey}
                  onResolve={onResolve}
                />
              </section>

              <section className="insightPanel">
                <div className="insightPanelHeader">
                  <strong>Low human audit score</strong>
                  <span>Human review disagreed or found issues</span>
                </div>
                <InsightQuestionRows
                  items={insights.low_human_audit_score_questions}
                  mode="human"
                  empty="No low human audit score questions found."
                  resolvingKey={resolvingKey}
                  onResolve={onResolve}
                />
              </section>
            </div>
          </>
        )}
      </section>
    </section>
  );
}

function humanAuditReason(response: ChatResponse) {
  const reason = response.human_audit_prompt?.reason;
  if (reason === 'low_score') return 'Requested because the LLM-as-judge score was low.';
  if (reason === 'interval') return `Requested as part of the scheduled review sample.`;
  if (reason === 'interval_and_low_score') {
    return 'Requested because this response is in the review sample and the LLM-as-judge score was low.';
  }
  return 'Requested as part of the human audit sample.';
}

function HumanAuditForm({
  response,
  submitting,
  onSubmit,
  onDismiss,
}: {
  response: ChatResponse;
  submitting: boolean;
  onSubmit: (payload: HumanAuditPayload) => Promise<void>;
  onDismiss: () => void;
}) {
  const [scores, setScores] = useState<Partial<Record<AuditScoreKey, number>>>({});
  const [overallVerdict, setOverallVerdict] = useState<'pass' | 'partial' | 'fail' | 'unable_to_judge'>('partial');
  const [issueTags, setIssueTags] = useState<string[]>([]);
  const [comment, setComment] = useState('');
  const [reviewerConfidence, setReviewerConfidence] = useState<'low' | 'medium' | 'high'>('medium');
  const complete = AUDIT_DIMENSIONS.every((dimension) => scores[dimension.key]);

  function toggleIssue(value: string) {
    setIssueTags((current) => (
      current.includes(value)
        ? current.filter((item) => item !== value)
        : [...current, value]
    ));
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    event.stopPropagation();
    if (!complete) return;
    await onSubmit({
      chat_log_id: response.chat_log_id,
      groundedness_score: scores.groundedness_score ?? 1,
      citation_score: scores.citation_score ?? 1,
      relevance_score: scores.relevance_score ?? 1,
      completeness_score: scores.completeness_score ?? 1,
      clarity_score: scores.clarity_score ?? 1,
      overall_verdict: overallVerdict,
      issue_tags: issueTags,
      comment: comment.trim() || undefined,
      reviewer_confidence: reviewerConfidence,
    });
  }

  return (
    <form className="humanAuditForm" onSubmit={submit} onClick={(event) => event.stopPropagation()}>
      <div className="humanAuditHeader">
        <div>
          <strong>Human review requested</strong>
          <span>{humanAuditReason(response)}</span>
        </div>
        <button type="button" onClick={onDismiss} disabled={submitting}>
          Skip
        </button>
      </div>
      <div className="auditScoreRows">
        {AUDIT_DIMENSIONS.map((dimension) => (
          <div className="auditScoreRow" key={dimension.key}>
            <div>
              <strong>{dimension.label}</strong>
              <span>{dimension.help}</span>
            </div>
            <div className="scoreButtons" role="radiogroup" aria-label={dimension.label}>
              {[1, 2, 3, 4, 5].map((score) => (
                <button
                  type="button"
                  key={score}
                  className={scores[dimension.key] === score ? 'selected' : ''}
                  onClick={() => setScores((current) => ({ ...current, [dimension.key]: score }))}
                >
                  {score}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="auditGrid">
        <label className="auditSelectField compact">
          Overall verdict
          <select value={overallVerdict} onChange={(event) => setOverallVerdict(event.target.value as 'pass' | 'partial' | 'fail' | 'unable_to_judge')}>
            <option value="pass">Pass</option>
            <option value="partial">Partial</option>
            <option value="fail">Fail</option>
            <option value="unable_to_judge">Unable to judge</option>
          </select>
        </label>
        <label className="auditSelectField">
          Reviewer confidence
          <select value={reviewerConfidence} onChange={(event) => setReviewerConfidence(event.target.value as 'low' | 'medium' | 'high')}>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
          <em>
            How sure are you that your review is accurate? Choose low if you were unsure about the
            policy or did not check citations; choose high if you checked the evidence carefully.
          </em>
        </label>
      </div>
      <div className="auditIssueTags">
        {AUDIT_ISSUES.map((issue) => (
          <label key={issue.value} className={issueTags.includes(issue.value) ? 'selected' : ''}>
            <input
              type="checkbox"
              checked={issueTags.includes(issue.value)}
              onChange={() => toggleIssue(issue.value)}
            />
            {issue.label}
          </label>
        ))}
      </div>
      <label className="auditComment">
        Optional comment
        <textarea
          value={comment}
          onChange={(event) => setComment(event.target.value)}
          placeholder="What should be improved, or why did you score it this way?"
          rows={3}
        />
      </label>
      <div className="humanAuditActions">
        <button type="submit" disabled={submitting || !complete}>
          {submitting ? 'Saving review...' : 'Submit human review'}
        </button>
      </div>
    </form>
  );
}

function JudgeScoreTile({ label, score }: { label: string; score?: number }) {
  const level = answerQualityLevel(score);
  const width = score === undefined ? 0 : Math.max(0, Math.min(100, (score / 5) * 100));
  return (
    <div className={`judgeScoreTile ${level}`}>
      <span>{label}</span>
      <strong>{answerQualityDisplay(score)}</strong>
      <div className="judgeMeter" aria-hidden="true">
        <i style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}

function LlmJudgePanel({
  response,
  onBack,
  onClose,
}: {
  response: ChatResponse;
  onBack: () => void;
  onClose: () => void;
}) {
  const overall = answerQualityScore(response);
  const verdict = judgeString(response, 'verdict') ?? 'not judged';
  const confidence = judgeString(response, 'confidence') ?? 'unknown';
  const notes = judgeString(response, 'notes');
  const failureModes = judgeList(response, 'failure_modes');
  const improvements = judgeList(response, 'required_improvements');
  const unsupportedClaims = judgeList(response, 'unsupported_claims');
  const missingCitations = judgeList(response, 'missing_citations');
  const incorrectCitations = judgeList(response, 'incorrect_citations');
  const claimChecks = judgeClaimChecks(response);

  return (
    <>
      <div className="panelHeader">
        <button className="judgeBackButton" onClick={onBack}>
          <ChevronLeft size={17} />
          Evidence
        </button>
        <div className="panelHeaderActions">
          <Sparkles size={18} />
          <button className="iconButton panelCollapseButton" onClick={onClose} title="Hide LLM judge">
            <ChevronRight size={18} />
          </button>
        </div>
      </div>

      <div className="judgeHero">
        <span>LLM-as-judge review</span>
        <strong>{answerQualityDisplay(overall)}</strong>
        <div className="judgeHeroMeta">
          <em className={`judgeVerdict ${verdict}`}>{verdict}</em>
          <em>{confidence} confidence</em>
        </div>
      </div>

      <div className="judgeScoreGrid">
        <JudgeScoreTile label="Groundedness" score={judgeNumber(response, 'groundedness_score')} />
        <JudgeScoreTile label="Citation" score={judgeNumber(response, 'citation_score')} />
        <JudgeScoreTile label="Relevance" score={judgeNumber(response, 'relevance_score')} />
        <JudgeScoreTile label="Completeness" score={judgeNumber(response, 'completeness_score')} />
        <JudgeScoreTile label="Clarity" score={judgeNumber(response, 'clarity_score')} />
      </div>

      {notes && (
        <section className="judgeSection">
          <h3>Judge Notes</h3>
          <p>{notes}</p>
        </section>
      )}

      {failureModes.length > 0 && (
        <section className="judgeSection">
          <h3>Failure Modes</h3>
          <div className="judgeChipList">
            {failureModes.map((item) => (
              <span key={item}>{item.replace(/_/g, ' ')}</span>
            ))}
          </div>
        </section>
      )}

      {(improvements.length > 0 || unsupportedClaims.length > 0 || missingCitations.length > 0 || incorrectCitations.length > 0) && (
        <section className="judgeSection">
          <h3>Review Flags</h3>
          <ul className="judgeList">
            {improvements.map((item) => <li key={`improvement-${item}`}>{item}</li>)}
            {unsupportedClaims.map((item) => <li key={`unsupported-${item}`}>Unsupported: {item}</li>)}
            {missingCitations.map((item) => <li key={`missing-${item}`}>Missing citation: {item}</li>)}
            {incorrectCitations.map((item) => <li key={`incorrect-${item}`}>Incorrect citation: {item}</li>)}
          </ul>
        </section>
      )}

      {claimChecks.length > 0 && (
        <section className="judgeSection">
          <h3>Claim Checks</h3>
          <div className="claimCheckList">
            {claimChecks.slice(0, 6).map((item, index) => (
              <article key={`${item.claim}-${index}`}>
                <strong>{item.claim}</strong>
                <span>{item.support}</span>
                <em>{item.citations}</em>
                {item.issue && <p>{item.issue}</p>}
              </article>
            ))}
          </div>
        </section>
      )}

      <details className="judgeRawDetails">
        <summary>Raw judge JSON</summary>
        <pre className="judgeBox">{JSON.stringify(response.judge, null, 2)}</pre>
      </details>
    </>
  );
}

function bytesToKb(bytes: number) {
  return `${Math.round(bytes / 1024).toLocaleString()} KB`;
}

function formatRecentDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value));
}

function historyItemToResponse(item: ChatHistoryItem): ChatResponse {
  return {
    chat_log_id: item.id,
    session_id: item.session_id,
    question: item.question,
    answer: item.answer,
    chunking_strategy: item.chunking_strategy,
    citations: item.citations,
    evidence: item.evidence ?? [],
    judge: item.judge ?? null,
    retrieval_score: item.retrieval_score,
    refused: item.refused,
    top_k: item.top_k,
    run_judge: item.run_judge,
    retrieved_count: item.retrieved_count,
    corpus_version: item.corpus_version,
    index_version: item.index_version,
    latency_ms: {
      embedding: item.embedding_latency_ms,
      vector_query: item.vector_query_latency_ms,
      generation: item.generation_latency_ms,
      judge: item.judge_latency_ms,
      total: item.total_latency_ms,
    },
    models: {
      embedding: item.embedding_model,
      generation: item.generation_model,
      judge: item.judge_model,
    },
  };
}

function LoginScreen({ onLogin }: { onLogin: (user: User) => void }) {
  const [email, setEmail] = useState('demo@example.com');
  const [password, setPassword] = useState('demo1234');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError('');
    try {
      const result = await login(email, password);
      setToken(result.access_token);
      onLogin(result.user);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="loginShell">
      <section className="loginPanel">
        <div className="loginBrand">
          <div className="loginBrandMark">K</div>
          <div>
            <strong>Kraken Policy Assistant</strong>
            <span>Academic MVP</span>
          </div>
        </div>
        <h1>Sign in to continue</h1>
        <p>Ask questions over the policy corpus with citation-grounded answers.</p>
        <form onSubmit={submit} className="loginForm">
          <label>
            Email
            <input value={email} onChange={(event) => setEmail(event.target.value)} type="email" />
          </label>
          <label>
            Password
            <input value={password} onChange={(event) => setPassword(event.target.value)} type="password" />
          </label>
          {error && <div className="formError">{error}</div>}
          <button className="primaryButton" disabled={loading}>
            {loading ? <Loader2 className="spin" size={18} /> : <Sparkles size={18} />}
            Sign in
          </button>
        </form>
      </section>
    </main>
  );
}

function StatusPill({ ok, children }: { ok: boolean; children: React.ReactNode }) {
  return <span className={ok ? 'statusPill ok' : 'statusPill warn'}>{children}</span>;
}

function AssistantAvatar() {
  return (
    <div className="assistantAvatar" aria-hidden="true">
      K
    </div>
  );
}

function UsagePolicyModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="modalBackdrop" role="dialog" aria-modal="true" aria-labelledby="usage-policy-title">
      <section className="policyModal">
        <div className="modalHeader">
          <div>
            <p className="eyebrow">Usage policy</p>
            <h2 id="usage-policy-title">Terms of Use for the Kraken Policy Assistant</h2>
          </div>
          <button className="iconButton" onClick={onClose} aria-label="Close usage policy">
            Close
          </button>
        </div>
        <div className="policyBody">
          <p>
            This virtual assistant has been developed for an MSc dissertation project to support testing of a
            retrieval-augmented chatbot over selected Kraken policy documents.
          </p>
          <h3>1. Purpose</h3>
          <p>
            The assistant is intended to answer questions using the uploaded policy document corpus. It is a research
            prototype and should be used for general information and user testing only.
          </p>
          <h3>2. Accuracy and limitations</h3>
          <p>
            The assistant may produce incorrect, incomplete, or outdated responses. It can also misunderstand questions
            or cite evidence imperfectly. You should verify important information against the cited source documents.
          </p>
          <h3>3. No professional advice</h3>
          <p>
            Responses do not constitute legal, financial, investment, regulatory, or professional advice. Do not rely on
            the assistant to make financial or compliance decisions.
          </p>
          <h3>4. Responsible use</h3>
          <p>
            Please use the assistant for questions related to the document corpus. Do not submit abusive, illegal,
            confidential, personal, financial, or sensitive information.
          </p>
          <h3>5. Data and feedback</h3>
          <p>
            Questions, answers, ratings, comments, timestamps, and system metadata may be stored for dissertation
            evaluation. Feedback submitted through thumbs up/down may be used to improve and evaluate the prototype.
          </p>
          <h3>6. Source documents</h3>
          <p>
            The assistant answers from the configured policy corpus only. If the relevant information is not present in
            those documents, the assistant may refuse or provide an incomplete answer.
          </p>
        </div>
        <button className="primaryButton modalAction" onClick={onClose}>
          I understand
        </button>
      </section>
    </div>
  );
}

type DemoFeedbackFormState = Omit<
  DemoFeedbackPayload,
  'ease_score' | 'helpfulness_score' | 'trustworthiness_score'
> & {
  ease_score: number | '';
  helpfulness_score: number | '';
  trustworthiness_score: number | '';
};

const initialDemoFeedback: DemoFeedbackFormState = {
  role_department: '',
  ease_score: '',
  helpfulness_score: '',
  trustworthiness_score: '',
  citations_useful: 'somewhat',
  incorrect_or_misleading: false,
  incorrect_notes: '',
  missing_expectation: '',
  improvement: '',
  would_use_at_work: 'maybe',
  final_comments: '',
};

function DemoFeedbackModal({ onClose }: { onClose: () => void }) {
  const [form, setForm] = useState<DemoFeedbackFormState>(initialDemoFeedback);
  const [submitting, setSubmitting] = useState(false);
  const [status, setStatus] = useState('');

  function update<K extends keyof DemoFeedbackFormState>(key: K, value: DemoFeedbackFormState[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function optionalText(value?: string) {
    const trimmed = value?.trim();
    return trimmed ? trimmed : undefined;
  }

  const canSubmit =
    form.ease_score !== '' && form.helpfulness_score !== '' && form.trustworthiness_score !== '';

  async function submitFeedback(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit) {
      setStatus('Please score ease, helpfulness, and trustworthiness.');
      return;
    }

    setSubmitting(true);
    setStatus('');
    try {
      const easeScore = Number(form.ease_score);
      const helpfulnessScore = Number(form.helpfulness_score);
      const trustworthinessScore = Number(form.trustworthiness_score);
      await sendDemoFeedback({
        role_department: optionalText(form.role_department),
        ease_score: easeScore,
        helpfulness_score: helpfulnessScore,
        trustworthiness_score: trustworthinessScore,
        citations_useful: form.citations_useful,
        incorrect_or_misleading: Boolean(optionalText(form.incorrect_notes)),
        incorrect_notes: optionalText(form.incorrect_notes),
        missing_expectation: optionalText(form.missing_expectation),
        improvement: optionalText(form.improvement),
        would_use_at_work: form.would_use_at_work,
        final_comments: optionalText(form.final_comments),
      });
      setStatus('Thank you, your demo feedback was saved.');
      setForm(initialDemoFeedback);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : 'Could not save demo feedback');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="modalBackdrop demoFeedbackOverlay" role="dialog" aria-modal="true" aria-labelledby="demo-feedback-title">
      <form className="policyModal demoFeedbackModal" onSubmit={submitFeedback}>
        <div className="modalHeader">
          <div>
            <p className="eyebrow">Post-demo questionnaire</p>
            <h2 id="demo-feedback-title">Demo feedback</h2>
            <span>Short feedback for the dissertation evaluation.</span>
          </div>
          <button type="button" className="iconButton" onClick={onClose} aria-label="Close demo feedback">
            Close
          </button>
        </div>

        <div className="demoFeedbackBody">
          <label>
            Role or general department <span>optional</span>
            <input
              value={form.role_department ?? ''}
              onChange={(event) => update('role_department', event.target.value)}
              placeholder="Compliance, Legal, Support, Product, Other"
            />
          </label>

          <div className="demoScoreGrid">
            <label>
              Ease of use
              <select
                value={form.ease_score}
                onChange={(event) => update('ease_score', Number(event.target.value))}
                required
              >
                <option value="">Select</option>
                <option value={1}>1 - Very difficult</option>
                <option value={2}>2 - Difficult</option>
                <option value={3}>3 - Neutral</option>
                <option value={4}>4 - Easy</option>
                <option value={5}>5 - Very easy</option>
              </select>
            </label>
            <label>
              Helpfulness
              <select
                value={form.helpfulness_score}
                onChange={(event) => update('helpfulness_score', Number(event.target.value))}
                required
              >
                <option value="">Select</option>
                <option value={1}>1 - Not helpful</option>
                <option value={2}>2 - Slightly helpful</option>
                <option value={3}>3 - Neutral</option>
                <option value={4}>4 - Helpful</option>
                <option value={5}>5 - Very helpful</option>
              </select>
            </label>
            <label>
              Trustworthiness
              <select
                value={form.trustworthiness_score}
                onChange={(event) => update('trustworthiness_score', Number(event.target.value))}
                required
              >
                <option value="">Select</option>
                <option value={1}>1 - Not trustworthy</option>
                <option value={2}>2 - Slightly trustworthy</option>
                <option value={3}>3 - Neutral</option>
                <option value={4}>4 - Trustworthy</option>
                <option value={5}>5 - Very trustworthy</option>
              </select>
            </label>
          </div>

          <div className="demoCompactGrid">
            <label>
              Were citations useful?
              <select
                value={form.citations_useful}
                onChange={(event) => update('citations_useful', event.target.value as DemoFeedbackPayload['citations_useful'])}
              >
                <option value="yes">Yes</option>
                <option value="somewhat">Somewhat</option>
                <option value="no">No</option>
              </select>
            </label>
            <label>
              Would you use this at work?
              <select
                value={form.would_use_at_work}
                onChange={(event) => update('would_use_at_work', event.target.value as DemoFeedbackPayload['would_use_at_work'])}
              >
                <option value="yes">Yes</option>
                <option value="maybe">Maybe</option>
                <option value="no">No</option>
              </select>
            </label>
          </div>

          <label>
            Was there an answer that seemed incorrect, incomplete, or misleading, and why?
            <textarea
              value={form.incorrect_notes ?? ''}
              onChange={(event) => update('incorrect_notes', event.target.value)}
              rows={3}
              placeholder="Optional"
            />
          </label>

          <label>
            Was there anything you expected it to answer but it could not?
            <textarea
              value={form.missing_expectation ?? ''}
              onChange={(event) => update('missing_expectation', event.target.value)}
              rows={3}
            />
          </label>

          <label>
            What is one thing you would improve?
            <textarea
              value={form.improvement ?? ''}
              onChange={(event) => update('improvement', event.target.value)}
              rows={3}
            />
          </label>

          <label>
            Any final comments?
            <textarea
              value={form.final_comments ?? ''}
              onChange={(event) => update('final_comments', event.target.value)}
              rows={3}
            />
          </label>

          {status && <p className="feedbackStatus">{status}</p>}
        </div>

        <div className="demoFeedbackActions">
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" disabled={submitting || !canSubmit}>
            {submitting ? 'Saving...' : 'Submit feedback'}
          </button>
        </div>
      </form>
    </div>
  );
}

export function App() {
  const [user, setUser] = useState<User | null>(null);
  const [booting, setBooting] = useState(true);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [webDocuments, setWebDocuments] = useState<DocumentItem[]>([]);
  const [adminMetrics, setAdminMetrics] = useState<AdminMetrics | null>(null);
  const [adminInsights, setAdminInsights] = useState<AdminInsights | null>(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [showResolvedInsights, setShowResolvedInsights] = useState(false);
  const [resolvingInsightKey, setResolvingInsightKey] = useState<string | null>(null);
  const [strategy, setStrategy] = useState<Strategy>('structure_aware');
  const [topK, setTopK] = useState(5);
  const [runJudge, setRunJudge] = useState(false);
  const [metadataRerank, setMetadataRerank] = useState(true);
  const [humanAuditEnabled, setHumanAuditEnabled] = useState(false);
  const [humanAuditInterval, setHumanAuditInterval] = useState(5);
  const [humanAuditLowScoreThreshold, setHumanAuditLowScoreThreshold] = useState(3);
  const [humanAuditMaxPerSession, setHumanAuditMaxPerSession] = useState(1);
  const [humanAuditCooldownMinutes, setHumanAuditCooldownMinutes] = useState(10);
  const [retrievalSettings, setRetrievalSettings] = useState<RetrievalSettings | null>(null);
  const [question, setQuestion] = useState('');
  const [conversation, setConversation] = useState<ConversationItem[]>([]);
  const [recentChats, setRecentChats] = useState<ChatHistorySession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null);
  const [activeResponse, setActiveResponse] = useState<ChatResponse | null>(null);
  const [loadingAnswer, setLoadingAnswer] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [crawlingWeb, setCrawlingWeb] = useState(false);
  const [detectingWebDuplicates, setDetectingWebDuplicates] = useState(false);
  const [webDuplicateReport, setWebDuplicateReport] = useState<WebDuplicateReport | null>(null);
  const [webSeedUrls, setWebSeedUrls] = useState('https://www.kraken.com/legal\nhttps://www.kraken.com/legal/disclosures');
  const [webMaxPages, setWebMaxPages] = useState(20);
  const [webMaxDepth, setWebMaxDepth] = useState(1);
  const [deletingDocument, setDeletingDocument] = useState<string | null>(null);
  const [notice, setNotice] = useState('');
  const [feedbackMode, setFeedbackMode] = useState<'closed' | 'needs_work'>('closed');
  const [feedbackComment, setFeedbackComment] = useState('');
  const [feedbackStatus, setFeedbackStatus] = useState('');
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [submittingHumanAudit, setSubmittingHumanAudit] = useState<number | null>(null);
  const [submittedHumanAudits, setSubmittedHumanAudits] = useState<Record<number, boolean>>({});
  const [dismissedHumanAudits, setDismissedHumanAudits] = useState<Record<number, boolean>>({});
  const [showUsagePolicy, setShowUsagePolicy] = useState(false);
  const [showDemoFeedback, setShowDemoFeedback] = useState(false);
  const [showRecentChatsPanel, setShowRecentChatsPanel] = useState(false);
  const [showAdminDashboardPanel, setShowAdminDashboardPanel] = useState(false);
  const [showEvidencePanel, setShowEvidencePanel] = useState(true);
  const [rightPanelView, setRightPanelView] = useState<'evidence' | 'judge'>('evidence');
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [centerView, setCenterView] = useState<'chat' | 'corpus' | 'insights'>('chat');
  const [corpusTypeFilter, setCorpusTypeFilter] = useState<'all' | 'pdf' | 'web'>('all');
  const [corpusSort, setCorpusSort] = useState<'name' | 'pdf_first' | 'web_first' | 'size'>('name');

  const indexed = useMemo(() => {
    if (!status) return false;
    return (status.index_counts.fixed_size ?? 0) > 0 && (status.index_counts.structure_aware ?? 0) > 0;
  }, [status]);

  const isAdmin = user?.role === 'admin';
  const isTesterUser = Boolean(user && /^tester\d{2}@example\.com$/i.test(user.email));
  const showRecentChats = showRecentChatsPanel;
  const showAdminDashboard = isAdmin && showAdminDashboardPanel;
  const showEvidence = showEvidencePanel;
  const corpusItems = useMemo(() => {
    const items = [
      ...documents.map((doc) => ({ ...doc, source_type: 'pdf' as const })),
      ...webDocuments.map((doc) => ({ ...doc, source_type: 'web' as const })),
    ].filter((doc) => corpusTypeFilter === 'all' || doc.source_type === corpusTypeFilter);

    return items.sort((a, b) => {
      if (corpusSort === 'pdf_first') {
        return `${a.source_type === 'pdf' ? 0 : 1}-${a.file_name}`.localeCompare(
          `${b.source_type === 'pdf' ? 0 : 1}-${b.file_name}`,
        );
      }
      if (corpusSort === 'web_first') {
        return `${a.source_type === 'web' ? 0 : 1}-${a.file_name}`.localeCompare(
          `${b.source_type === 'web' ? 0 : 1}-${b.file_name}`,
        );
      }
      if (corpusSort === 'size') {
        return b.bytes - a.bytes;
      }
      return a.file_name.localeCompare(b.file_name);
    });
  }, [corpusSort, corpusTypeFilter, documents, webDocuments]);

  async function refreshAdminData() {
    const [statusResult, docsResult, metricsResult] = await Promise.all([
      getStatus(),
      getDocuments(),
      getAdminMetrics(),
    ]);
    setStatus(statusResult);
    setDocuments(docsResult.documents);
    setWebDocuments(docsResult.web_documents ?? []);
    setAdminMetrics(metricsResult);
  }

  async function refreshAdminInsights(includeResolved = showResolvedInsights) {
    setInsightsLoading(true);
    try {
      const result = await getAdminInsights(includeResolved);
      setAdminInsights(result);
      setNotice('');
    } catch (err) {
      setNotice(err instanceof Error ? err.message : 'Could not load insights');
    } finally {
      setInsightsLoading(false);
    }
  }

  async function handleShowResolvedInsights(value: boolean) {
    setShowResolvedInsights(value);
    await refreshAdminInsights(value);
  }

  async function handleResolveInsight(issueType: InsightIssueType, itemKey: string, resolved: boolean) {
    const key = `${issueType}:${itemKey}`;
    setResolvingInsightKey(key);
    try {
      await resolveAdminInsight({ issue_type: issueType, item_key: itemKey, resolved });
      await refreshAdminInsights(showResolvedInsights);
      setNotice(resolved ? 'Insight flag marked as resolved.' : 'Insight flag reopened.');
    } catch (err) {
      setNotice(err instanceof Error ? err.message : 'Could not update insight flag');
    } finally {
      setResolvingInsightKey(null);
    }
  }

  async function refreshRecentChats() {
    const result = await getChatHistory();
    setRecentChats(result.items);
  }

  async function openRecentChatsPanel() {
    setShowUserMenu(false);
    setShowAdminDashboardPanel(false);
    setCenterView('chat');
    setShowRecentChatsPanel(true);
    await refreshRecentChats();
  }

  async function openCorpusManager() {
    setCenterView('corpus');
    setShowUserMenu(false);
    setShowRecentChatsPanel(false);
    setShowEvidencePanel(false);
    if (isAdmin) {
      await refreshAdminData();
    }
  }

  async function openInsightsDashboard() {
    setCenterView('insights');
    setShowUserMenu(false);
    setShowRecentChatsPanel(false);
    setShowAdminDashboardPanel(false);
    setShowEvidencePanel(false);
    if (isAdmin && !adminInsights) {
      await refreshAdminInsights();
    }
  }

  function applyRetrievalSettings(settings: RetrievalSettings) {
    setRetrievalSettings(settings);
    setStrategy(settings.chunking_strategy);
    setTopK(settings.top_k);
    setRunJudge(settings.run_judge);
    setMetadataRerank(settings.metadata_rerank_enabled);
    setHumanAuditEnabled(settings.human_audit_enabled);
    setHumanAuditInterval(settings.human_audit_interval);
    setHumanAuditLowScoreThreshold(settings.human_audit_low_score_threshold);
    setHumanAuditMaxPerSession(settings.human_audit_max_per_session);
    setHumanAuditCooldownMinutes(settings.human_audit_cooldown_minutes);
  }

  async function loadRetrievalSettings() {
    const settings = await getRetrievalSettings();
    applyRetrievalSettings(settings);
  }

  useEffect(() => {
    async function boot() {
      const token = getToken();
      if (!token) {
        setBooting(false);
        return;
      }
      try {
        const current = await me();
        setUser(current);
        setShowAdminDashboardPanel(current.role === 'admin');
        await loadRetrievalSettings();
        if (current.role === 'admin') {
          await refreshAdminData();
        } else {
          await refreshRecentChats();
        }
      } catch {
        clearToken();
      } finally {
        setBooting(false);
      }
    }
    boot();
  }, []);

  useEffect(() => {
    setFeedbackMode('closed');
    setFeedbackComment('');
    setFeedbackStatus('');
    setRightPanelView('evidence');
  }, [activeResponse?.chat_log_id]);

  async function handleLogin(nextUser: User) {
    setUser(nextUser);
    setShowAdminDashboardPanel(nextUser.role === 'admin');
    setShowEvidencePanel(true);
    await loadRetrievalSettings();
    if (nextUser.role === 'admin') {
      await refreshAdminData();
    } else {
      await refreshRecentChats();
    }
  }

  async function handleRetrievalSettingsUpdate(update: Partial<RetrievalSettings>) {
    if (!retrievalSettings) return;
    const nextSettings = { ...retrievalSettings, ...update };
    applyRetrievalSettings(nextSettings);
    setNotice('Saving retrieval setup...');
    try {
      const saved = await updateRetrievalSettings(nextSettings);
      applyRetrievalSettings(saved);
      setNotice('Retrieval setup saved. User accounts will use this configuration.');
    } catch (err) {
      setNotice(err instanceof Error ? err.message : 'Could not save retrieval setup');
      applyRetrievalSettings(retrievalSettings);
    }
  }

  async function handleIngest(force = false) {
    setIndexing(true);
    setNotice('Indexing documents. This can take a few minutes because embeddings are generated for both chunking strategies.');
    try {
      const result = await ingest(force);
      setNotice(`Indexed ${result.counts.fixed_size} fixed chunks and ${result.counts.structure_aware} structure-aware chunks.`);
      await refreshAdminData();
    } catch (err) {
      setNotice(err instanceof Error ? err.message : 'Indexing failed');
    } finally {
      setIndexing(false);
    }
  }

  async function handleUpload(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    setNotice('Uploading policy documents. Rebuild indexes after upload so the chatbot can retrieve them.');
    try {
      const result = await uploadDocuments(Array.from(files));
      setNotice(`Uploaded ${result.uploaded.length} document(s). Rebuild indexes before testing the new corpus.`);
      await refreshAdminData();
    } catch (err) {
      setNotice(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  }

  async function handleWebCrawl() {
    const seed_urls = webSeedUrls
      .split(/\r?\n/)
      .map((url) => url.trim())
      .filter(Boolean);
    if (seed_urls.length === 0) {
      setNotice('Add at least one seed URL before crawling.');
      return;
    }
    setCrawlingWeb(true);
    setNotice('Crawling public web pages. Rebuild indexes after the crawl finishes.');
    try {
      const result = await crawlWebCorpus({
        seed_urls,
        max_pages: webMaxPages,
        max_depth: webMaxDepth,
      });
      setNotice(
        `Saved ${result.pages_saved} web page(s), visited ${result.visited}. Rebuild indexes to include web content.`,
      );
      setWebDuplicateReport(null);
      await refreshAdminData();
    } catch (err) {
      setNotice(err instanceof Error ? err.message : 'Web crawl failed');
    } finally {
      setCrawlingWeb(false);
    }
  }

  async function handleDetectWebDuplicates() {
    setDetectingWebDuplicates(true);
    setNotice('Checking crawled web pages for duplicate or near-duplicate content...');
    try {
      const result = await detectWebDuplicates();
      setWebDuplicateReport(result);
      setNotice(
        result.duplicate_group_count > 0
          ? `Found ${result.duplicate_group_count} duplicate or near-duplicate group(s) across ${result.total_pages} web page(s).`
          : `No duplicate or near-duplicate pages found across ${result.total_pages} web page(s).`,
      );
    } catch (err) {
      setNotice(err instanceof Error ? err.message : 'Duplicate detection failed');
    } finally {
      setDetectingWebDuplicates(false);
    }
  }

  async function handleDeleteDocument(fileName: string) {
    const confirmed = window.confirm(
      `Remove "${fileName}" from the corpus and indexes? You can upload it again later if needed.`,
    );
    if (!confirmed) return;
    setDeletingDocument(fileName);
    setNotice(`Removing ${fileName} from the corpus...`);
    try {
      const result = await deleteDocument(fileName);
      const deletedChunks = Object.values(result.vector_chunks_deleted).reduce((total, count) => total + count, 0);
      setNotice(`Removed ${fileName} and ${deletedChunks} indexed chunks. Corpus v${result.corpus_version}, index v${result.index_version}.`);
      await refreshAdminData();
    } catch (err) {
      setNotice(err instanceof Error ? err.message : 'Document could not be removed');
    } finally {
      setDeletingDocument(null);
    }
  }

  async function handleDeleteWebDocument(doc: DocumentItem) {
    if (!doc.url) {
      setNotice('This web corpus item is missing its URL, so it cannot be removed.');
      return;
    }
    const confirmed = window.confirm(
      `Remove "${doc.file_name}" from the web corpus and indexes? You can crawl it again later if needed.`,
    );
    if (!confirmed) return;
    setDeletingDocument(doc.url);
    setNotice(`Removing ${doc.file_name} from the web corpus...`);
    try {
      const result = await deleteWebCorpusItem(doc.url);
      const deletedChunks = Object.values(result.vector_chunks_deleted).reduce((total, count) => total + count, 0);
      setNotice(
        `Removed ${doc.file_name} and ${deletedChunks} indexed chunks. Corpus v${result.corpus_version}, index v${result.index_version}.`,
      );
      setWebDuplicateReport(null);
      await refreshAdminData();
    } catch (err) {
      setNotice(err instanceof Error ? err.message : 'Web corpus item could not be removed');
    } finally {
      setDeletingDocument(null);
    }
  }

  async function submitQuestion(text: string) {
    const trimmed = text.trim();
    if (!trimmed || loadingAnswer) return;
    setLoadingAnswer(true);
    setQuestion('');
    const userItem: ConversationItem = { id: crypto.randomUUID(), role: 'user', text: trimmed };
    setConversation((items) => [...items, userItem]);

    try {
      const response = await askQuestion(trimmed, activeSessionId);
      setActiveResponse(response);
      setActiveSessionId(response.session_id);
      setShowEvidencePanel(true);
      setRightPanelView('evidence');
      setConversation((items) => [
        ...items,
        { id: crypto.randomUUID(), role: 'assistant', text: response.answer, response },
      ]);
      await refreshRecentChats();
      if (isAdmin) await refreshAdminData();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'The request failed';
      setConversation((items) => [...items, { id: crypto.randomUUID(), role: 'assistant', text: message }]);
    } finally {
      setLoadingAnswer(false);
    }
  }

  async function handleAsk(event: React.FormEvent) {
    event.preventDefault();
    await submitQuestion(question);
  }

  function logout() {
    clearToken();
    setUser(null);
    setStatus(null);
    setDocuments([]);
    setRetrievalSettings(null);
    setConversation([]);
    setRecentChats([]);
    setActiveSessionId(null);
    setActiveResponse(null);
    setShowRecentChatsPanel(false);
    setShowAdminDashboardPanel(false);
    setShowEvidencePanel(true);
    setShowDemoFeedback(false);
    setRightPanelView('evidence');
    setShowUserMenu(false);
    setCenterView('chat');
    setAdminInsights(null);
    setSubmittedHumanAudits({});
    setDismissedHumanAudits({});
  }

  function resetConversation() {
    setConversation([]);
    setActiveResponse(null);
    setActiveSessionId(null);
    setQuestion('');
    setFeedbackMode('closed');
    setFeedbackComment('');
    setFeedbackStatus('');
    setShowEvidencePanel(true);
    setRightPanelView('evidence');
    setSubmittedHumanAudits({});
    setDismissedHumanAudits({});
  }

  function handleNewChat() {
    resetConversation();
    setCenterView('chat');
    setShowRecentChatsPanel(false);
    setShowUserMenu(false);
  }

  async function openRecentChat(item: ChatHistorySession) {
    setCenterView('chat');
    const result = await getChatSession(item.id);
    const restoredConversation = result.items.flatMap((historyItem) => {
      const response = historyItemToResponse(historyItem);
      return [
        { id: `history-question-${historyItem.id}`, role: 'user' as const, text: historyItem.question },
        { id: `history-answer-${historyItem.id}`, role: 'assistant' as const, text: historyItem.answer, response },
      ];
    });
    const responseItems = restoredConversation.filter((entry) => entry.role === 'assistant' && entry.response);
    const lastResponse = responseItems.length > 0 ? responseItems[responseItems.length - 1].response ?? null : null;
    setActiveResponse(lastResponse);
    setActiveSessionId(item.id);
    setShowEvidencePanel(true);
    setRightPanelView('evidence');
    setConversation(restoredConversation);
  }

  async function handleDeleteRecentChat(sessionId: number) {
    await deleteChatHistoryItem(sessionId);
    setRecentChats((items) => items.filter((item) => item.id !== sessionId));
    if (activeSessionId === sessionId) {
      resetConversation();
    }
  }

  async function handleClearRecentChats() {
    await clearChatHistory();
    setRecentChats([]);
    resetConversation();
  }

  function handleCitationOpen(
    response: ChatResponse,
    citation: ChatResponse['citations'][number] | ChatResponse['evidence'][number],
  ) {
    trackCitationClick({
      chat_log_id: response.chat_log_id,
      citation_label: citation.label,
      chunk_id: citation.chunk_id,
      file_name: citation.file_name,
      page_start: citation.page_start,
      page_end: citation.page_end,
      source_strategy: citation.source_strategy,
    }).catch(() => undefined);
  }

  async function handleFeedback(rating: number, feedbackType: 'helpful' | 'needs_work', comment?: string) {
    if (!activeResponse || submittingFeedback) return;
    setSubmittingFeedback(true);
    setFeedbackStatus('');
    try {
      await sendFeedback(activeResponse.chat_log_id, rating, comment, feedbackType);
      setFeedbackStatus(rating >= 4 ? 'Thanks, feedback saved.' : 'Thanks, your note was saved.');
      setFeedbackMode('closed');
      setFeedbackComment('');
    } catch (err) {
      setFeedbackStatus(err instanceof Error ? err.message : 'Feedback could not be saved.');
    } finally {
      setSubmittingFeedback(false);
    }
  }

  async function handleHumanAudit(payload: HumanAuditPayload) {
    setSubmittingHumanAudit(payload.chat_log_id);
    try {
      await sendHumanAudit(payload);
      setSubmittedHumanAudits((current) => ({ ...current, [payload.chat_log_id]: true }));
    } catch (err) {
      setFeedbackStatus(err instanceof Error ? err.message : 'Could not save human review');
    } finally {
      setSubmittingHumanAudit(null);
    }
  }

  function dismissHumanAudit(chatLogId: number) {
    setDismissedHumanAudits((current) => ({ ...current, [chatLogId]: true }));
  }

  const centerTitle =
    isAdmin && centerView === 'corpus'
      ? 'Manage corpus'
      : isAdmin && centerView === 'insights'
        ? 'Insights dashboard'
        : 'Kraken Policy Assistant';
  const centerSubtitle =
    isAdmin && centerView === 'corpus'
      ? 'Source documents and web crawl setup'
      : isAdmin && centerView === 'insights'
        ? 'User testing and corpus gap signals'
        : isAdmin
          ? 'Admin workspace'
          : 'Academic MVP';

  if (booting) {
    return (
      <main className="boot">
        <Loader2 className="spin" />
      </main>
    );
  }

  if (!user) {
    return <LoginScreen onLogin={handleLogin} />;
  }

  return (
    <main
      className={`appShell krakenShell ${isAdmin ? 'adminShell' : 'userShell'} ${
        showAdminDashboard ? 'withAdminDashboard' : ''
      } ${showRecentChats ? 'withRecentChats' : ''} ${showEvidence ? 'withEvidence' : ''}`}
    >
      {showUsagePolicy && <UsagePolicyModal onClose={() => setShowUsagePolicy(false)} />}
      {showDemoFeedback && <DemoFeedbackModal onClose={() => setShowDemoFeedback(false)} />}
      {showAdminDashboard && (
      <aside className="sidebar adminDashboardPanel">
        <div className="sidebarTop">
          <div className="productLockup">
            <div className="krakenMark smallMark">K</div>
            <div>
              <strong>Admin dashboard</strong>
              <span>Corpus and retrieval setup</span>
            </div>
          </div>
          <button className="iconButton panelCollapseButton" onClick={() => setShowAdminDashboardPanel(false)} title="Hide admin dashboard">
            <ChevronLeft size={18} />
          </button>
        </div>

        <section className="controlSection">
          <div className="sectionTitle">
            <Database size={17} />
            Corpus Status
          </div>
          <div className="statusStack">
            <StatusPill ok={Boolean(status?.pdf_dir_exists)}>
              {status?.pdf_dir_exists ? 'PDF folder found' : 'PDF folder missing'}
            </StatusPill>
            <StatusPill ok={Boolean(status?.openai_key_configured)}>
              {status?.openai_key_configured ? 'OpenAI key ready' : 'OpenAI key missing'}
            </StatusPill>
            <StatusPill ok={indexed}>{indexed ? 'Indexes ready' : 'Needs indexing'}</StatusPill>
          </div>
          <div className="countGrid">
            <span>Fixed</span>
            <strong>{status?.index_counts.fixed_size ?? 0}</strong>
            <span>Structure</span>
            <strong>{status?.index_counts.structure_aware ?? 0}</strong>
            <span>Web pages</span>
            <strong>{status?.web_pages_count ?? 0}</strong>
            <span>Corpus v</span>
            <strong>{status?.corpus_version ?? 1}</strong>
            <span>Index v</span>
            <strong>{status?.index_version ?? 1}</strong>
          </div>
          <button className="secondaryButton" disabled={indexing} onClick={() => handleIngest(!indexed)}>
            {indexing ? <Loader2 className="spin" size={17} /> : <RefreshCw size={17} />}
            {indexed ? 'Refresh indexes' : 'Build indexes'}
          </button>
          {notice && <p className="notice">{notice}</p>}
        </section>

        <section className="controlSection">
          <div className="sectionTitle">
            <Search size={17} />
            Retrieval Setup
          </div>
          <div className="segmented">
            <button
              className={strategy === 'structure_aware' ? 'selected' : ''}
              onClick={() => handleRetrievalSettingsUpdate({ chunking_strategy: 'structure_aware' })}
            >
              Structure-aware
            </button>
            <button
              className={strategy === 'fixed_size' ? 'selected' : ''}
              onClick={() => handleRetrievalSettingsUpdate({ chunking_strategy: 'fixed_size' })}
            >
              Fixed-size
            </button>
            <button
              className={strategy === 'hybrid' ? 'selected' : ''}
              onClick={() => handleRetrievalSettingsUpdate({ chunking_strategy: 'hybrid' })}
            >
              Hybrid
            </button>
          </div>
          <p className="strategyHint">
            Hybrid queries both chunk indexes, removes near-duplicates, then reranks with document and section metadata.
          </p>
          <label className="rangeLabel">
            Top K
            <input
              type="range"
              min="1"
              max="10"
              value={topK}
              onChange={(event) => handleRetrievalSettingsUpdate({ top_k: Number(event.target.value) })}
            />
            <strong>{topK}</strong>
          </label>
          <label className="switchRow">
            <input
              type="checkbox"
              checked={runJudge}
              onChange={(event) => handleRetrievalSettingsUpdate({ run_judge: event.target.checked })}
            />
            LLM-as-judge validation
          </label>
          <label className="switchRow">
            <input
              type="checkbox"
              checked={metadataRerank}
              onChange={(event) => handleRetrievalSettingsUpdate({ metadata_rerank_enabled: event.target.checked })}
            />
            Metadata reranking
          </label>
          <div className="humanAuditSettings">
            <label className="switchRow">
              <input
                type="checkbox"
                checked={humanAuditEnabled}
                onChange={(event) => handleRetrievalSettingsUpdate({ human_audit_enabled: event.target.checked })}
              />
              Human audit prompts
            </label>
            <p className="strategyHint">
              Prompts users to score selected answers so you can compare human review with LLM-as-judge results.
            </p>
            <div className="auditSettingsGrid">
              <label>
                Every X queries
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={humanAuditInterval}
                  onChange={(event) => handleRetrievalSettingsUpdate({ human_audit_interval: Number(event.target.value) })}
                />
                <span>Use 0 to disable scheduled sampling.</span>
              </label>
              <label>
                Low LLM score
                <input
                  type="number"
                  min="1"
                  max="5"
                  value={humanAuditLowScoreThreshold}
                  onChange={(event) => handleRetrievalSettingsUpdate({ human_audit_low_score_threshold: Number(event.target.value) })}
                />
                <span>Prompts when overall judge score is at or below this value.</span>
              </label>
              <label>
                Max per session
                <input
                  type="number"
                  min="0"
                  max="20"
                  value={humanAuditMaxPerSession}
                  onChange={(event) => handleRetrievalSettingsUpdate({ human_audit_max_per_session: Number(event.target.value) })}
                />
                <span>Caps how many human reviews appear in one chat. Use 0 for no cap.</span>
              </label>
              <label>
                Cooldown minutes
                <input
                  type="number"
                  min="0"
                  max="1440"
                  value={humanAuditCooldownMinutes}
                  onChange={(event) => handleRetrievalSettingsUpdate({ human_audit_cooldown_minutes: Number(event.target.value) })}
                />
                <span>Waits before prompting the same user again. Use 0 to disable.</span>
              </label>
            </div>
          </div>
        </section>

        <section className="controlSection documentsList">
          <div className="sectionTitle">
            <BookOpen size={17} />
            Corpus
          </div>
          <div className="corpusLauncherCard">
            <span>
              {documents.length} PDFs | {webDocuments.length} web pages
            </span>
            <strong>Manage source documents, crawled pages, uploads, and indexing inputs in the central workspace.</strong>
            <button className="secondaryButton" onClick={openCorpusManager}>
              <BookOpen size={17} />
              Manage corpus
            </button>
          </div>
        </section>

        <section className="controlSection adminMetricsSection">
          <div className="sectionTitle">
            <LayoutDashboard size={17} />
            Evaluation Snapshot
          </div>
          <div className="corpusLauncherCard insightLauncherCard">
            <span>Product insights</span>
            <strong>Review top question themes, refusals, weak judge scores, human audit issues, and corpus gap signals.</strong>
            <button className="secondaryButton" onClick={openInsightsDashboard}>
              <BarChart3 size={17} />
              Open insights
            </button>
          </div>
          <div className="adminMetricGrid">
            <span>
              <em>Total chats</em>
              <strong>{adminMetrics?.total_chats ?? 0}</strong>
            </span>
            <span>
              <em>Active users</em>
              <strong>{adminMetrics?.active_users ?? 0}</strong>
            </span>
            <span>
              <em>Human reviews</em>
              <strong>{adminMetrics?.total_human_audits ?? 0}</strong>
            </span>
            <span>
              <em>Citation opens</em>
              <strong>{adminMetrics?.citation_clicks ?? 0}</strong>
            </span>
            <span>
              <em>Avg human</em>
              <strong>{metricScoreDisplay(adminMetrics?.avg_human_score)}</strong>
            </span>
            <span>
              <em>Avg LLM</em>
              <strong>{metricScoreDisplay(adminMetrics?.avg_llm_score_snapshot)}</strong>
            </span>
          </div>
          <div className="verdictBreakdown">
            <span>Pass {adminMetrics?.human_verdicts.pass ?? 0}</span>
            <span>Partial {adminMetrics?.human_verdicts.partial ?? 0}</span>
            <span>Fail {adminMetrics?.human_verdicts.fail ?? 0}</span>
            <span>Unable {adminMetrics?.human_verdicts.unable_to_judge ?? 0}</span>
          </div>
          <p className="strategyHint">
            Prompted reviews: {adminMetrics?.human_audit_prompts ?? 0} | Refusals: {adminMetrics?.refusals ?? 0} | Avg retrieval:{' '}
            {scorePercent(adminMetrics?.avg_retrieval_score ?? undefined)}
          </p>
        </section>
      </aside>
      )}

      {showRecentChats && (
        <aside className="recentChatsPanel">
          <div className="recentHeader">
            <div>
              <h2>Recent chats</h2>
            </div>
            <div className="recentActions">
              <button className="historyIconButton" onClick={() => setShowRecentChatsPanel(false)} title="Hide recent chats">
                <ChevronLeft size={15} />
              </button>
              <button className="historyActionButton" onClick={refreshRecentChats} title="Refresh recent chats">
                <RefreshCw size={15} />
                Refresh
              </button>
              {recentChats.length > 0 && (
                <button className="historyIconButton danger" onClick={handleClearRecentChats} title="Clear recent chats">
                  <Trash2 size={15} />
                </button>
              )}
            </div>
          </div>
          <div className="recentChatList">
            {recentChats.slice(0, 12).map((item) => (
              <div
                key={item.id}
                className={activeSessionId === item.id ? 'recentChatItem active' : 'recentChatItem'}
              >
                <button className="recentChatOpen" onClick={() => openRecentChat(item)}>
                  <span>{item.title}</span>
                  <em>
                    {item.message_count} {item.message_count === 1 ? 'question' : 'questions'} |{' '}
                    {formatRecentDate(item.updated_at)}
                  </em>
                </button>
                <button
                  className="historyIconButton"
                  onClick={() => handleDeleteRecentChat(item.id)}
                  title="Delete this chat"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
            {recentChats.length === 0 && <p className="recentEmpty">No recent chats yet.</p>}
          </div>
        </aside>
      )}

      <section className="chatColumn">
        <header className="chatHeader">
          <div className="krakenLockup">
            <div className="krakenMark">K</div>
            <div>
              <strong>{centerTitle}</strong>
              <span>{centerSubtitle}</span>
            </div>
          </div>
          {isTesterUser && (
            <div className="headerCenterAction">
              <button className="demoFeedbackButton" onClick={() => setShowDemoFeedback(true)}>
                <FileQuestion size={17} />
                Demo feedback
              </button>
            </div>
          )}
          <div className="headerActions">
            <div className="userBadge">
              <CheckCircle2 size={16} />
              {user.email}
              {isAdmin && <span>Admin</span>}
            </div>
            <div className="userMenu">
              <button
                className="menuTrigger"
                onClick={() => setShowUserMenu((open) => !open)}
                aria-haspopup="menu"
                aria-expanded={showUserMenu}
              >
                <MoreVertical size={18} />
              </button>
              {showUserMenu && (
                <div className="menuContent" role="menu">
                  <button onClick={handleNewChat} role="menuitem">
                    <Plus size={16} />
                    New chat
                  </button>
                  <button onClick={openRecentChatsPanel} role="menuitem">
                    <MessageSquareText size={16} />
                    Recent chats
                  </button>
                  {isAdmin && (
                    <button onClick={openCorpusManager} role="menuitem">
                      <BookOpen size={16} />
                      Manage corpus
                    </button>
                  )}
                  {isAdmin && (
                    <button onClick={openInsightsDashboard} role="menuitem">
                      <BarChart3 size={16} />
                      Insights dashboard
                    </button>
                  )}
                  {isAdmin && (
                    <button
                      onClick={() => {
                        setShowAdminDashboardPanel(true);
                        setCenterView('chat');
                        setShowRecentChatsPanel(false);
                        setShowUserMenu(false);
                      }}
                      role="menuitem"
                    >
                      <LayoutDashboard size={16} />
                      Admin dashboard
                    </button>
                  )}
                  <button onClick={logout} role="menuitem">
                    <LogOut size={16} />
                    Sign out
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>

        {isAdmin && centerView === 'corpus' ? (
          <section className="corpusManager">
            <aside className="corpusManagerSide">
              <div className="corpusManagerTitle">
                <BookOpen size={18} />
                <div>
                  <strong>Manage corpus</strong>
                  <span>{documents.length + webDocuments.length} total sources</span>
                </div>
              </div>

              <div className="corpusControlBlock">
                <strong>Corpus type</strong>
                <div className="segmented corpusSegmented">
                  <button className={corpusTypeFilter === 'all' ? 'selected' : ''} onClick={() => setCorpusTypeFilter('all')}>
                    All
                  </button>
                  <button className={corpusTypeFilter === 'pdf' ? 'selected' : ''} onClick={() => setCorpusTypeFilter('pdf')}>
                    PDF
                  </button>
                  <button className={corpusTypeFilter === 'web' ? 'selected' : ''} onClick={() => setCorpusTypeFilter('web')}>
                    Web
                  </button>
                </div>
                <label>
                  Sort by
                  <select value={corpusSort} onChange={(event) => setCorpusSort(event.target.value as 'name' | 'pdf_first' | 'web_first' | 'size')}>
                    <option value="name">Name</option>
                    <option value="pdf_first">PDF first</option>
                    <option value="web_first">Web first</option>
                    <option value="size">Largest first</option>
                  </select>
                </label>
              </div>

              <div className="corpusControlBlock">
                <strong>Upload PDFs</strong>
                <label className="uploadDrop">
                  <input
                    type="file"
                    accept="application/pdf,.pdf"
                    multiple
                    onChange={(event) => {
                      handleUpload(event.target.files);
                      event.target.value = '';
                    }}
                    disabled={uploading}
                  />
                  {uploading ? 'Uploading...' : 'Upload PDF documents'}
                </label>
              </div>

              <div className="webCrawlerCard">
                <strong>Web scraper</strong>
                <span>Add public Kraken pages, then rebuild indexes so they can be retrieved.</span>
                <label>
                  Seed URLs
                  <textarea
                    value={webSeedUrls}
                    onChange={(event) => setWebSeedUrls(event.target.value)}
                    rows={4}
                    placeholder="https://www.kraken.com/legal"
                  />
                </label>
                <div className="webCrawlerGrid">
                  <label>
                    Max pages
                    <input
                      type="number"
                      min="1"
                      max="100"
                      value={webMaxPages}
                      onChange={(event) => setWebMaxPages(Number(event.target.value))}
                    />
                  </label>
                  <label>
                    Depth
                    <input
                      type="number"
                      min="0"
                      max="3"
                      value={webMaxDepth}
                      onChange={(event) => setWebMaxDepth(Number(event.target.value))}
                    />
                  </label>
                </div>
                <button className="secondaryButton" disabled={crawlingWeb} onClick={handleWebCrawl}>
                  {crawlingWeb ? <Loader2 className="spin" size={17} /> : <RefreshCw size={17} />}
                  {crawlingWeb ? 'Crawling...' : 'Crawl web pages'}
                </button>
                <button className="secondaryButton" disabled={detectingWebDuplicates || webDocuments.length === 0} onClick={handleDetectWebDuplicates}>
                  {detectingWebDuplicates ? <Loader2 className="spin" size={17} /> : <Search size={17} />}
                  {detectingWebDuplicates ? 'Checking...' : 'Detect duplicate pages'}
                </button>
                {webDuplicateReport && (
                  <div className="duplicateReport">
                    <strong>
                      {webDuplicateReport.duplicate_group_count} duplicate group{webDuplicateReport.duplicate_group_count === 1 ? '' : 's'}
                    </strong>
                    <span>
                      Checked {webDuplicateReport.total_pages} web page{webDuplicateReport.total_pages === 1 ? '' : 's'} at{' '}
                      {Math.round(webDuplicateReport.similarity_threshold * 100)}% near-match threshold.
                    </span>
                    {webDuplicateReport.duplicate_groups.slice(0, 5).map((group, groupIndex) => (
                      <article className="duplicateGroup" key={`${group.type}-${groupIndex}`}>
                        <em>
                          {group.type === 'exact' ? 'Exact duplicate' : 'Near duplicate'} | {Math.round(group.similarity * 100)}% similar
                        </em>
                        {group.pages.map((page) => (
                          <a href={page.url} target="_blank" rel="noreferrer" key={page.url}>
                            <span>{page.title}</span>
                            <small>{page.url}</small>
                          </a>
                        ))}
                      </article>
                    ))}
                    {webDuplicateReport.duplicate_groups.length > 5 && (
                      <span>Showing first 5 groups. Use the corpus list to inspect or remove pages.</span>
                    )}
                  </div>
                )}
              </div>
            </aside>

            <section className="corpusManagerMain">
              <div className="corpusHero">
                <div>
                  <span>Admin corpus workspace</span>
                  <h1>Manage corpus</h1>
                  <p>
                    Review PDFs and crawled web pages before rebuilding the retrieval indexes. Keep this corpus tight:
                    only sources that help answer realistic policy questions should stay here.
                  </p>
                </div>
                <button className="secondaryButton compactAction" disabled={indexing} onClick={() => handleIngest(!indexed)}>
                  {indexing ? <Loader2 className="spin" size={17} /> : <RefreshCw size={17} />}
                  {indexed ? 'Refresh indexes' : 'Build indexes'}
                </button>
              </div>

              <div className="corpusStatsRow">
                <span>
                  <em>PDFs</em>
                  <strong>{documents.length}</strong>
                </span>
                <span>
                  <em>Web pages</em>
                  <strong>{webDocuments.length}</strong>
                </span>
                <span>
                  <em>Fixed chunks</em>
                  <strong>{status?.index_counts.fixed_size ?? 0}</strong>
                </span>
                <span>
                  <em>Structure chunks</em>
                  <strong>{status?.index_counts.structure_aware ?? 0}</strong>
                </span>
              </div>

              {notice && <p className="notice">{notice}</p>}

              <div className="corpusListHeader">
                <strong>Sources</strong>
                <span>
                  Showing {corpusItems.length} {corpusTypeFilter === 'all' ? 'items' : `${corpusTypeFilter} items`}
                </span>
              </div>

              <div className="corpusSourceList">
                {corpusItems.map((doc) => {
                  const isWeb = doc.source_type === 'web';
                  const deletingKey = isWeb ? doc.url : doc.file_name;
                  return (
                    <div className="corpusSourceRow" key={doc.url ?? doc.file_name}>
                      <a
                        className="corpusSourceLink"
                        href={isWeb ? doc.url : getDocumentUrl(doc.file_name)}
                        target={isWeb ? '_blank' : undefined}
                        rel={isWeb ? 'noreferrer' : undefined}
                        download={isWeb ? undefined : doc.file_name}
                      >
                        <FileText size={16} />
                        <span>
                          <strong>{doc.file_name}</strong>
                          <em>{isWeb ? doc.url : 'PDF document'}</em>
                        </span>
                        <small>{isWeb ? 'web' : 'pdf'}</small>
                        <small>{bytesToKb(doc.bytes)}</small>
                        <Download size={15} />
                      </a>
                      <button
                        className="docDeleteButton"
                        onClick={() => (isWeb ? handleDeleteWebDocument(doc) : handleDeleteDocument(doc.file_name))}
                        disabled={deletingDocument === deletingKey}
                        title={`Remove ${doc.file_name} from corpus`}
                      >
                        {deletingDocument === deletingKey ? <Loader2 className="spin" size={14} /> : <Trash2 size={14} />}
                      </button>
                    </div>
                  );
                })}
                {corpusItems.length === 0 && (
                  <div className="corpusEmpty">
                    <BookOpen size={22} />
                    <strong>No sources match this filter.</strong>
                    <span>Try switching corpus type or upload/crawl a source.</span>
                  </div>
                )}
              </div>
            </section>
          </section>
        ) : isAdmin && centerView === 'insights' ? (
          <AdminInsightsDashboard
            insights={adminInsights}
            loading={insightsLoading}
            notice={notice}
            onRefresh={refreshAdminInsights}
            showResolved={showResolvedInsights}
            onShowResolvedChange={handleShowResolvedInsights}
            resolvingKey={resolvingInsightKey}
            onResolve={handleResolveInsight}
          />
        ) : (
          <>
        <section className="conversation">
          {conversation.length === 0 && (
            <>
              <article className="message assistant welcomeMessage">
                <AssistantAvatar />
                <div className="messageBody">
                  <div className="messageLabel">Kraken Policy Assistant</div>
                  <p>
                    Hi,
                    {'\n'}Lovely to meet you! I'm your virtual assistant.
                    {'\n\n'}I'm still learning, so I might not always get things right. Help me improve by rating my
                    responses with a thumbs up or down.
                  </p>
                </div>
              </article>
              <article className="message assistant promptMessage continuationMessage">
                <div className="messageBody">
                  <p>How can I help you today?</p>
                </div>
              </article>
              <div className="starterPanel">
                <div className="suggestedQuestions">
                  {SUGGESTED_QUESTIONS.map((suggestedQuestion) => (
                    <button
                      key={suggestedQuestion}
                      onClick={() => submitQuestion(suggestedQuestion)}
                      disabled={loadingAnswer}
                    >
                      <Sparkles size={16} />
                      <span>{suggestedQuestion}</span>
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
          {conversation.map((item, index) => {
            const showAssistantAvatar = item.role === 'assistant' && (index === 0 || conversation[index - 1].role !== 'assistant');
            return (
            <article
              className={`message ${item.role} ${item.role === 'assistant' && !showAssistantAvatar ? 'continuationMessage' : ''}`}
              key={item.id}
              onClick={() => {
                if (item.response) {
                  setActiveResponse(item.response);
                  setShowEvidencePanel(true);
                  setRightPanelView('evidence');
                }
              }}
            >
              {showAssistantAvatar && <AssistantAvatar />}
              <div className="messageBody">
              <div className="messageLabel">{item.role === 'user' ? 'You' : 'Kraken Policy Assistant'}</div>
              <p>{item.text}</p>
              {item.response && (
                <div className="answerMeta">
                  <AnswerQualityPill response={item.response} />
                  {isAdmin && <span>{strategyLabel(item.response.chunking_strategy)}</span>}
                  {isAdmin && <span>Retrieval {scorePercent(item.response.retrieval_score)}</span>}
                  {item.response.refused && <span className="warningText">refused</span>}
                </div>
              )}
              {item.response && !isAdmin && (
                <div className="compactSources">
                  {item.response.citations.slice(0, 3).map((citation) => (
                    <a
                      key={citation.chunk_id}
                      href={getCitationUrl(citation)}
                      target="_blank"
                      rel="noreferrer"
                      onClick={() => handleCitationOpen(item.response!, citation)}
                    >
                      {citation.label}: {citation.file_name}
                      {citation.source_path?.startsWith('http') ? ' | web' : `, p. ${citation.page_start}`}
                    </a>
                  ))}
                </div>
              )}
              {item.response && !isAdmin && activeResponse?.chat_log_id === item.response.chat_log_id && (
                <div className="inlineFeedback" onClick={(event) => event.stopPropagation()}>
                  <button disabled={submittingFeedback} onClick={() => handleFeedback(5, 'helpful')}>
                    <ThumbsUp size={16} />
                    <span>Helpful</span>
                  </button>
                  <button disabled={submittingFeedback} onClick={() => setFeedbackMode('needs_work')}>
                    <ThumbsDown size={16} />
                    <span>Needs work</span>
                  </button>
                  {feedbackMode === 'needs_work' && (
                    <form
                      className="feedbackForm compact"
                      onSubmit={(event) => {
                        event.preventDefault();
                        handleFeedback(2, 'needs_work', feedbackComment.trim());
                      }}
                    >
                      <textarea
                        value={feedbackComment}
                        onChange={(event) => setFeedbackComment(event.target.value)}
                        placeholder="What needs work?"
                        rows={3}
                      />
                      <button type="submit" disabled={submittingFeedback || !feedbackComment.trim()}>
                        Save note
                      </button>
                    </form>
                  )}
                  {feedbackStatus && <p className="feedbackStatus">{feedbackStatus}</p>}
                </div>
              )}
              {item.response?.human_audit_prompt?.show
                && !submittedHumanAudits[item.response.chat_log_id]
                && !dismissedHumanAudits[item.response.chat_log_id] && (
                  <HumanAuditForm
                    response={item.response}
                    submitting={submittingHumanAudit === item.response.chat_log_id}
                    onSubmit={handleHumanAudit}
                    onDismiss={() => dismissHumanAudit(item.response?.chat_log_id ?? 0)}
                  />
                )}
              </div>
            </article>
            );
          })}
          {loadingAnswer && (
            <article
              className={`message assistant typingMessage ${
                conversation.length > 0 && conversation[conversation.length - 1].role === 'assistant'
                  ? 'continuationMessage'
                  : ''
              }`}
            >
              {conversation.length === 0 || conversation[conversation.length - 1].role !== 'assistant' ? (
                <AssistantAvatar />
              ) : null}
              <div className="messageBody">
                <div className="messageLabel">Kraken Policy Assistant</div>
                <p>
                  I'm checking the policy documents
                  <span className="typingDots" aria-hidden="true">
                    <span />
                    <span />
                    <span />
                  </span>
                </p>
              </div>
            </article>
          )}
        </section>

        <form className="askBar" onSubmit={handleAsk}>
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Type your message here..."
          />
          <button className="primaryButton sendButton" disabled={loadingAnswer || !question.trim()} aria-label="Send message">
            <Send size={20} />
          </button>
        </form>
        {!isAdmin && (
          <footer className="chatDisclaimer">
            This AI chatbot can make mistakes. Please see our usage policy{' '}
            <a
              href="#usage-policy"
              onClick={(event) => {
                event.preventDefault();
                setShowUsagePolicy(true);
              }}
            >
              here
            </a>
            .
          </footer>
        )}
          </>
        )}
      </section>

      {showEvidence && (
      <aside className="evidencePanel">
        {rightPanelView === 'judge' && activeResponse ? (
          <LlmJudgePanel
            response={activeResponse}
            onBack={() => setRightPanelView('evidence')}
            onClose={() => setShowEvidencePanel(false)}
          />
        ) : (
        <>
        <div className="panelHeader">
          <h2>Evidence</h2>
          <div className="panelHeaderActions">
            {activeResponse?.refused ? <AlertCircle size={18} /> : <ShieldCheck size={18} />}
            <button className="iconButton panelCollapseButton" onClick={() => setShowEvidencePanel(false)} title="Hide evidence">
              <ChevronRight size={18} />
            </button>
          </div>
        </div>
        {!activeResponse && <p className="muted">Select an answer to inspect citations and retrieved chunks.</p>}
        {activeResponse && (
          <>
            <div className="metricCard">
              <span>Answer quality score</span>
              <strong>{answerQualityDisplay(answerQualityScore(activeResponse))}</strong>
              <em>{answerQualityLabel(answerQualityScore(activeResponse))}</em>
              <AnswerQualityPill response={activeResponse} />
              <button
                className="judgeToggleButton"
                disabled={!activeResponse.judge}
                onClick={() => setRightPanelView('judge')}
              >
                <Sparkles size={15} />
                {activeResponse.judge ? 'View LLM judge' : 'LLM judge not run'}
              </button>
            </div>
            <div className="metricCard">
              <span>Experiment settings</span>
              <strong>{strategyLabel(activeResponse.chunking_strategy)}</strong>
              <em>
                top K {activeResponse.top_k ?? topK}
                {activeResponse.latency_ms?.total ? ` | ${activeResponse.latency_ms.total} ms` : ''}
              </em>
            </div>
            <div className="citationList">
              {activeResponse.evidence.length > 0 ? activeResponse.evidence.map((item) => (
                <details key={item.chunk_id} open={item.label === 'C1'}>
                  <summary>
                    <span>{item.label}</span>
                    <strong>{item.section_title}</strong>
                  </summary>
                  <p className="citationSource">
                    {item.file_name}, pages {item.page_start}-{item.page_end}
                    {item.source_strategy ? ` | ${strategyLabel(item.source_strategy)}` : ''}
                  </p>
                  <div className="chunkScoreGrid">
                    <span>
                      Retrieval match
                      <strong>{scorePercent(item.score)}</strong>
                    </span>
                    <span>
                      Vector
                      <strong>{scorePercent(item.vector_score ?? undefined)}</strong>
                    </span>
                    <span>
                      Metadata boost
                      <strong>{boostLabel(item.metadata_boost)}</strong>
                    </span>
                  </div>
                  <a
                    className="documentLink"
                    href={getCitationUrl(item)}
                    target="_blank"
                    rel="noreferrer"
                    onClick={() => handleCitationOpen(activeResponse, item)}
                  >
                    {item.source_path?.startsWith('http') ? 'Open source page' : `Open PDF at page ${item.page_start}`}
                  </a>
                  <p className="chunkText">{item.text}</p>
                </details>
              )) : activeResponse.citations.map((item) => (
                <details key={item.chunk_id} open={item.label === 'C1'}>
                  <summary>
                    <span>{item.label}</span>
                    <strong>{item.section_title}</strong>
                  </summary>
                  <p className="citationSource">
                    {item.file_name}, pages {item.page_start}-{item.page_end}
                    {item.source_strategy ? ` | ${strategyLabel(item.source_strategy)}` : ''}
                  </p>
                  <div className="chunkScoreGrid">
                    <span>
                      Retrieval match
                      <strong>{scorePercent(item.score)}</strong>
                    </span>
                    <span>
                      Vector
                      <strong>{scorePercent(item.vector_score ?? undefined)}</strong>
                    </span>
                    <span>
                      Metadata boost
                      <strong>{boostLabel(item.metadata_boost)}</strong>
                    </span>
                  </div>
                  <a
                    className="documentLink"
                    href={getCitationUrl(item)}
                    target="_blank"
                    rel="noreferrer"
                    onClick={() => handleCitationOpen(activeResponse, item)}
                  >
                    {item.source_path?.startsWith('http') ? 'Open source page' : `Open PDF at page ${item.page_start}`}
                  </a>
                  <p className="chunkText muted">Excerpt was not stored for this older chat history item.</p>
                </details>
              ))}
            </div>
            {isAdmin && (
              <div className="feedbackRow">
                <button disabled={submittingFeedback} onClick={() => handleFeedback(5, 'helpful')}>
                  Helpful
                </button>
                <button disabled={submittingFeedback} onClick={() => setFeedbackMode('needs_work')}>
                  Needs work
                </button>
              </div>
            )}
            {isAdmin && feedbackMode === 'needs_work' && (
              <form
                className="feedbackForm"
                onSubmit={(event) => {
                  event.preventDefault();
                  handleFeedback(2, 'needs_work', feedbackComment.trim());
                }}
              >
                <label>
                  What needs work?
                  <textarea
                    value={feedbackComment}
                    onChange={(event) => setFeedbackComment(event.target.value)}
                    placeholder="Example: citation did not support the answer, answer was too vague, wrong document retrieved..."
                    rows={4}
                  />
                </label>
                <div className="feedbackActions">
                  <button type="button" onClick={() => setFeedbackMode('closed')} disabled={submittingFeedback}>
                    Cancel
                  </button>
                  <button type="submit" disabled={submittingFeedback || !feedbackComment.trim()}>
                    {submittingFeedback ? 'Saving...' : 'Save note'}
                  </button>
                </div>
              </form>
            )}
            {feedbackStatus && <p className="feedbackStatus">{feedbackStatus}</p>}
          </>
        )}
        </>
        )}
      </aside>
      )}
    </main>
  );
}
