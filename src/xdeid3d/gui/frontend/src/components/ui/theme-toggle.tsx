import { Moon, Sun, Monitor } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { useTheme } from '@/hooks/useTheme'

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" className="h-9 w-9">
          <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
          <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
          <span className="sr-only">Toggle theme</span>
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-40">
        <div className="flex flex-col gap-1">
          <Button
            variant={theme === 'light' ? 'secondary' : 'ghost'}
            size="sm"
            className="justify-start"
            onClick={() => setTheme('light')}
          >
            <Sun className="mr-2 h-4 w-4" />
            Light
          </Button>
          <Button
            variant={theme === 'dark' ? 'secondary' : 'ghost'}
            size="sm"
            className="justify-start"
            onClick={() => setTheme('dark')}
          >
            <Moon className="mr-2 h-4 w-4" />
            Dark
          </Button>
          <Button
            variant={theme === 'system' ? 'secondary' : 'ghost'}
            size="sm"
            className="justify-start"
            onClick={() => setTheme('system')}
          >
            <Monitor className="mr-2 h-4 w-4" />
            System
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  )
}
