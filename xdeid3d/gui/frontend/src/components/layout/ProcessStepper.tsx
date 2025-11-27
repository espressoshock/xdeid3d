import { Check } from 'lucide-react'
import { cn } from '@/lib/utils'
import { ThemeToggle } from '@/components/ui/theme-toggle'

interface Step {
  id: string
  title: string
  label: string
}

interface ProcessStepperProps {
  steps: Step[]
  currentStep: string
  onStepClick: (stepId: string) => void
  completedSteps: string[]
}

export function ProcessStepper({
  steps,
  currentStep,
  onStepClick,
  completedSteps
}: ProcessStepperProps) {
  const currentIndex = steps.findIndex(s => s.id === currentStep)

  return (
    <div className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-6 py-6 relative">
        <div className="flex items-center justify-center">
          <nav aria-label="Progress">
            <ol className="flex items-center gap-2">
              {steps.map((step, index) => {
                const isCompleted = completedSteps.includes(step.id)
                const isCurrent = step.id === currentStep
                const isDisabled = !isCompleted && !isCurrent
                const isClickable = isCompleted || isCurrent

                return (
                  <li key={step.id} className="flex items-center gap-2">
                    <button
                      onClick={() => isClickable && onStepClick(step.id)}
                      disabled={isDisabled}
                      className={cn(
                        "group flex items-center gap-3 rounded-lg px-4 py-3 transition-all",
                        isClickable && "cursor-pointer hover:bg-accent",
                        isDisabled && "cursor-not-allowed opacity-50"
                      )}
                    >
                      <div className="flex items-center gap-3">
                        {/* Step Circle */}
                        <div
                          className={cn(
                            "flex h-10 w-10 shrink-0 items-center justify-center rounded-full border-2 font-semibold transition-all",
                            isCurrent && "border-primary bg-primary text-primary-foreground shadow-lg shadow-primary/20",
                            isCompleted && "border-primary bg-primary text-primary-foreground",
                            isDisabled && "border-muted bg-muted text-muted-foreground"
                          )}
                        >
                          {isCompleted ? (
                            <Check className="h-5 w-5" />
                          ) : (
                            <span>{index + 1}</span>
                          )}
                        </div>

                        {/* Step Text */}
                        <div className="flex flex-col items-start">
                          <span className="text-xs font-medium text-muted-foreground">
                            {step.label}
                          </span>
                          <span
                            className={cn(
                              "text-sm font-semibold",
                              isCurrent && "text-foreground",
                              isCompleted && "text-foreground",
                              isDisabled && "text-muted-foreground"
                            )}
                          >
                            {step.title}
                          </span>
                        </div>
                      </div>
                    </button>

                    {/* Connector Line */}
                    {index < steps.length - 1 && (
                      <div
                        className={cn(
                          "h-0.5 w-12 transition-all",
                          index < currentIndex ? "bg-primary" : "bg-border"
                        )}
                      />
                    )}
                  </li>
                )
              })}
            </ol>
          </nav>
        </div>

        {/* Theme Toggle - Vertically centered with stepper */}
        <div className="absolute right-6 top-1/2 -translate-y-1/2">
          <ThemeToggle />
        </div>
      </div>
    </div>
  )
}
