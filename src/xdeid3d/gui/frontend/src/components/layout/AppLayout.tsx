import { useState, useCallback } from 'react'
import { ProcessStepper } from './ProcessStepper'
import { Tabs, TabsContent } from '@/components/ui/tabs'
import { GenerationPanel } from '../generation/GenerationPanel'
import { CustomizationPanel } from '../customization/CustomizationPanel'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { AlertCircle } from 'lucide-react'

const STEPS = [
  { id: 'generation', title: 'Generation', label: 'Step 1' },
  { id: 'customization', title: 'Customization', label: 'Step 2' },
  { id: 'audit', title: 'Audit & Results', label: 'Step 3' },
]

export function AppLayout() {
  const [activeTab, setActiveTab] = useState('generation')
  const [completedSteps, setCompletedSteps] = useState<string[]>([])

  const handleProceedToCustomization = useCallback(() => {
    setCompletedSteps(prev =>
      prev.includes('generation') ? prev : [...prev, 'generation']
    )
    setActiveTab('customization')
  }, [])

  const handleStepClick = useCallback((stepId: string) => {
    setActiveTab(stepId)
  }, [])

  return (
    <div className="flex flex-col h-screen">
      <ProcessStepper
        steps={STEPS}
        currentStep={activeTab}
        onStepClick={handleStepClick}
        completedSteps={completedSteps}
      />

      <main className="flex-1 overflow-hidden">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full">
          <TabsContent value="generation" className="h-full mt-0 p-0">
            <GenerationPanel onProceedToCustomization={handleProceedToCustomization} />
          </TabsContent>

          <TabsContent value="customization" className="h-full mt-0 p-0">
            <CustomizationPanel />
          </TabsContent>

          <TabsContent value="audit" className="h-full p-6">
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Coming Soon</AlertTitle>
              <AlertDescription>
                Audit & Visualization stage is under development. This will generate videos/frames,
                process them through the de-identification network, compute metrics, and display 3D heatmaps.
              </AlertDescription>
            </Alert>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
