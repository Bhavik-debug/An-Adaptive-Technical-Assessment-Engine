import { Plus, Trash2, X } from 'lucide-react';
import * as React from 'react';

import { ApiError } from '@/api/client';
import type { CandidateProfile, Evidence, ProfileSkill, TopicOption } from '@/api/types';
import { FormField } from '@/components/FormField';
import { Alert } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { keyLabel } from '@/lib/format';

/** backend/app/ingestion/profile.py limits. */
const MAX_SKILLS = 60;
const MAX_SUBTOPICS_PER_SKILL = 5;
const MAX_HEADLINE = 120;
const MAX_SKILL_NAME = 80;

const EVIDENCE_LABELS: Record<Evidence, string> = {
  listed: 'Listed on resume',
  project: 'Used in a project',
  professional: 'Used professionally',
};

interface SkillRow extends ProfileSkill {
  rowId: number;
}

let nextRowId = 1;
function toRows(skills: Partial<ProfileSkill>[] | undefined): SkillRow[] {
  return (skills ?? []).map((s) => ({
    rowId: nextRowId++,
    name: s.name ?? '',
    evidence: s.evidence && s.evidence in EVIDENCE_LABELS ? s.evidence : 'listed',
    subtopic_keys: (s.subtopic_keys ?? []).slice(0, MAX_SUBTOPICS_PER_SKILL),
  }));
}

interface Errors {
  headline?: string;
  years?: string;
  skills: Record<number, string>;
  form?: string;
}

export function ProfileEditor({
  initial,
  topics,
  confirmed,
  onConfirm,
}: {
  initial: CandidateProfile;
  topics: TopicOption[];
  confirmed: boolean;
  onConfirm: (profile: CandidateProfile) => Promise<void>;
}) {
  const [headline, setHeadline] = React.useState(initial.headline ?? '');
  const [years, setYears] = React.useState(
    initial.years_experience == null ? '' : String(initial.years_experience),
  );
  const [skills, setSkills] = React.useState<SkillRow[]>(() => toRows(initial.skills));
  const [errors, setErrors] = React.useState<Errors>({ skills: {} });
  const [saving, setSaving] = React.useState(false);
  const [dirty, setDirty] = React.useState(false);

  const subtopicNames = React.useMemo(
    () => new Map(topics.flatMap((t) => t.subtopics.map((s) => [s.key, s.display_name] as const))),
    [topics],
  );

  function edit<T>(setter: React.Dispatch<React.SetStateAction<T>>) {
    return (value: React.SetStateAction<T>) => {
      setter(value);
      setDirty(true);
    };
  }
  const setSkillsDirty = edit(setSkills);

  function updateSkill(rowId: number, patch: Partial<ProfileSkill>) {
    setSkillsDirty((rows) => rows.map((r) => (r.rowId === rowId ? { ...r, ...patch } : r)));
  }

  function validate(): Errors {
    const next: Errors = { skills: {} };
    if (headline.trim().length > MAX_HEADLINE)
      next.headline = `Keep the headline under ${MAX_HEADLINE} characters.`;
    if (years.trim()) {
      const n = Number(years);
      if (!Number.isFinite(n) || n < 0 || n > 60)
        next.years = 'Enter a number of years between 0 and 60.';
    }
    skills.forEach((s, index) => {
      if (!s.name.trim()) next.skills[index] = 'Enter a skill name or remove this row.';
      else if (s.name.trim().length > MAX_SKILL_NAME)
        next.skills[index] = `Keep skill names under ${MAX_SKILL_NAME} characters.`;
    });
    if (skills.length > MAX_SKILLS) next.form = `You can list up to ${MAX_SKILLS} skills.`;
    return next;
  }

  async function handleConfirm() {
    if (saving) return;
    const next = validate();
    setErrors(next);
    if (next.headline || next.years || next.form || Object.keys(next.skills).length) return;
    const profile: CandidateProfile = {
      headline: headline.trim() || null,
      years_experience: years.trim() ? Number(years) : null,
      skills: skills.map(({ name, evidence, subtopic_keys }) => ({
        name: name.trim(),
        evidence,
        subtopic_keys,
      })),
    };
    setSaving(true);
    try {
      await onConfirm(profile);
      setDirty(false);
    } catch (error) {
      const message = error instanceof ApiError ? error.message : 'Could not save your profile.';
      const skillErrors: Record<number, string> = {};
      if (error instanceof ApiError) {
        for (const [field, msg] of Object.entries(error.fieldErrors)) {
          const match = /^skills\.(\d+)/.exec(field);
          if (match) skillErrors[Number(match[1])] = msg;
        }
      }
      setErrors({ skills: skillErrors, form: message });
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-3">
        <FormField
          id="profile-headline"
          label="Headline"
          optional
          error={errors.headline}
          className="sm:col-span-2"
        >
          {(a11y) => (
            <Input
              {...a11y}
              value={headline}
              maxLength={MAX_HEADLINE}
              placeholder="e.g. Backend engineer"
              onChange={(e) => edit(setHeadline)(e.target.value)}
            />
          )}
        </FormField>
        <FormField id="profile-years" label="Years of experience" optional error={errors.years}>
          {(a11y) => (
            <Input
              {...a11y}
              type="number"
              inputMode="decimal"
              min={0}
              max={60}
              step="0.5"
              value={years}
              onChange={(e) => edit(setYears)(e.target.value)}
            />
          )}
        </FormField>
      </div>

      <fieldset className="space-y-3">
        <legend className="text-sm font-medium">Skills</legend>
        {skills.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No skills listed. Add any you would like the interview to take into account.
          </p>
        )}
        <ul className="space-y-3">
          {skills.map((skill, index) => {
            const nameId = `skill-${skill.rowId}-name`;
            const error = errors.skills[index];
            const available = topics.flatMap((t) =>
              t.subtopics
                .filter((s) => !skill.subtopic_keys.includes(s.key))
                .map((s) => ({ ...s, topic: t.display_name })),
            );
            return (
              <li key={skill.rowId} className="space-y-3 rounded-md border bg-muted/40 p-3">
                <div className="grid gap-3 sm:grid-cols-[1fr_auto_auto] sm:items-end">
                  <FormField id={nameId} label="Skill" error={error}>
                    {(a11y) => (
                      <Input
                        {...a11y}
                        value={skill.name}
                        maxLength={MAX_SKILL_NAME}
                        onChange={(e) => updateSkill(skill.rowId, { name: e.target.value })}
                      />
                    )}
                  </FormField>
                  <FormField id={`skill-${skill.rowId}-evidence`} label="Experience">
                    {(a11y) => (
                      <Select
                        {...a11y}
                        value={skill.evidence}
                        onChange={(e) =>
                          updateSkill(skill.rowId, { evidence: e.target.value as Evidence })
                        }
                      >
                        {Object.entries(EVIDENCE_LABELS).map(([value, label]) => (
                          <option key={value} value={value}>
                            {label}
                          </option>
                        ))}
                      </Select>
                    )}
                  </FormField>
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label={`Remove skill ${skill.name || index + 1}`}
                    onClick={() =>
                      setSkillsDirty((rows) => rows.filter((r) => r.rowId !== skill.rowId))
                    }
                  >
                    <Trash2 aria-hidden="true" />
                  </Button>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs text-muted-foreground">Related areas:</span>
                  {skill.subtopic_keys.length === 0 && (
                    <span className="text-xs text-muted-foreground">none</span>
                  )}
                  {skill.subtopic_keys.map((key) => (
                    <span
                      key={key}
                      className="inline-flex items-center gap-1 rounded-full border bg-card py-0.5 pl-2.5 pr-1 text-xs"
                    >
                      {keyLabel(key, subtopicNames)}
                      <button
                        type="button"
                        className="rounded-full p-0.5 hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        aria-label={`Remove ${keyLabel(key, subtopicNames)} from ${skill.name || 'this skill'}`}
                        onClick={() =>
                          updateSkill(skill.rowId, {
                            subtopic_keys: skill.subtopic_keys.filter((k) => k !== key),
                          })
                        }
                      >
                        <X className="size-3" aria-hidden="true" />
                      </button>
                    </span>
                  ))}
                  {skill.subtopic_keys.length < MAX_SUBTOPICS_PER_SKILL && available.length > 0 && (
                    <Select
                      aria-label={`Add a related area to ${skill.name || 'this skill'}`}
                      className="h-8 w-auto max-w-full py-1 text-xs"
                      value=""
                      onChange={(e) => {
                        if (e.target.value)
                          updateSkill(skill.rowId, {
                            subtopic_keys: [...skill.subtopic_keys, e.target.value],
                          });
                      }}
                    >
                      <option value="">Add area…</option>
                      {topics.map((topic) => (
                        <optgroup key={topic.key} label={topic.display_name}>
                          {available
                            .filter((s) => s.topic === topic.display_name)
                            .map((s) => (
                              <option key={s.key} value={s.key}>
                                {s.display_name}
                              </option>
                            ))}
                        </optgroup>
                      ))}
                    </Select>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
        {skills.length < MAX_SKILLS && (
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              setSkillsDirty((rows) => [...rows, ...toRows([{ name: '', evidence: 'listed' }])])
            }
          >
            <Plus aria-hidden="true" />
            Add skill
          </Button>
        )}
      </fieldset>

      {errors.form && (
        <Alert variant="error" title="Your profile was not saved">
          {errors.form}
        </Alert>
      )}
      {confirmed && !dirty && !errors.form && (
        <Alert variant="success" title="Profile confirmed">
          This profile will be used as a starting point for your interview. Your answers decide your
          results.
        </Alert>
      )}
      <div className="flex flex-wrap items-center gap-3">
        <Button
          onClick={handleConfirm}
          loading={saving}
          disabled={confirmed && !dirty}
          variant={confirmed && !dirty ? 'outline' : 'default'}
        >
          {confirmed ? (dirty ? 'Save changes' : 'Saved') : 'Confirm profile'}
        </Button>
        {dirty && <span className="text-xs text-muted-foreground">You have unsaved changes.</span>}
      </div>
    </div>
  );
}
