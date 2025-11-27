import { ThemeToggle } from '@/components/ui/theme-toggle'

export function Navigation() {
  return (
    <header className="border-b">
      <div className="flex h-16 items-center px-6">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/10 text-primary">
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            </svg>
          </div>
          <div>
            <h1 className="text-xl font-bold">X-DeID3D</h1>
            <p className="text-xs text-muted-foreground">Identity Auditing Platform</p>
          </div>
        </div>

        <div className="ml-auto flex items-center gap-4">
          <div className="text-sm text-muted-foreground">
            Explainable 3D De-identification Auditing
          </div>
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
