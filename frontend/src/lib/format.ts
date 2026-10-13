const dateTime = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'medium',
  timeStyle: 'short',
});

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return '';
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? '' : dateTime.format(date);
}

const LEVEL_LABELS: Record<string, string> = {
  junior: 'Junior',
  mid: 'Mid-level',
  senior: 'Senior',
  staff: 'Staff',
};

const MODE_LABELS: Record<string, string> = {
  screening: 'Screening',
  standard: 'Standard',
  deep_dive: 'Deep dive',
};

/** Fall back to a readable form of the key for values the UI has no label for. */
function humanise(key: string): string {
  const text = key.replace(/_/g, ' ');
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function levelLabel(level: string | null | undefined): string {
  return level ? (LEVEL_LABELS[level] ?? humanise(level)) : '';
}

export function modeLabel(mode: string | null | undefined): string {
  return mode ? (MODE_LABELS[mode] ?? humanise(mode)) : '';
}

export function keyLabel(key: string, names?: Map<string, string>): string {
  return names?.get(key) ?? humanise(key);
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
