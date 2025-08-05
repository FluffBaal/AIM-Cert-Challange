import * as React from "react"

interface CollapsibleProps {
  open?: boolean
  onOpenChange?: (open: boolean) => void
  children: React.ReactNode
}

interface CollapsibleTriggerProps {
  children: React.ReactNode
  className?: string
  asChild?: boolean
}

interface CollapsibleContentProps {
  children: React.ReactNode
  className?: string
}

const CollapsibleContext = React.createContext<{
  open: boolean
  onOpenChange: (open: boolean) => void
}>({
  open: false,
  onOpenChange: () => {},
})

const Collapsible = React.forwardRef<HTMLDivElement, CollapsibleProps>(
  ({ open = false, onOpenChange = () => {}, children }, ref) => {
    return (
      <CollapsibleContext.Provider value={{ open, onOpenChange }}>
        <div ref={ref}>{children}</div>
      </CollapsibleContext.Provider>
    )
  }
)
Collapsible.displayName = "Collapsible"

const CollapsibleTrigger = React.forwardRef<HTMLButtonElement, CollapsibleTriggerProps>(
  ({ children, className = "", asChild = false }, ref) => {
    const { open, onOpenChange } = React.useContext(CollapsibleContext)
    
    const handleClick = () => {
      onOpenChange(!open)
    }
    
    if (asChild && React.isValidElement(children)) {
      return React.cloneElement(children as React.ReactElement<any>, {
        onClick: handleClick,
        ref,
      })
    }
    
    return (
      <button
        ref={ref}
        className={className}
        onClick={handleClick}
        type="button"
      >
        {children}
      </button>
    )
  }
)
CollapsibleTrigger.displayName = "CollapsibleTrigger"

const CollapsibleContent = React.forwardRef<HTMLDivElement, CollapsibleContentProps>(
  ({ children, className = "" }, ref) => {
    const { open } = React.useContext(CollapsibleContext)
    
    if (!open) return null
    
    return (
      <div ref={ref} className={className}>
        {children}
      </div>
    )
  }
)
CollapsibleContent.displayName = "CollapsibleContent"

export { Collapsible, CollapsibleContent, CollapsibleTrigger }