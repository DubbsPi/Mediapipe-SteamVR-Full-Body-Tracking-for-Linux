import pygame
from PIL import Image

class SimpleImageDisplay:
    def __init__(self, width, height, caption="Image", fullscreen=True):
        pygame.init()
        
        self.width = width
        self.height = height
        
        # Create window
        if fullscreen:
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.NOFRAME)
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))
        
        pygame.display.set_caption(caption)
        
        self.current_image = None
        self.image_x = 0
        self.image_y = 0
        self.clock = pygame.time.Clock()
        
        blank = pygame.Surface((16, 16), pygame.SRCALPHA)  # fully transparent
        blank_cursor = pygame.cursors.Cursor((0, 0), blank)
        pygame.mouse.set_cursor(blank_cursor)

        print(f"Display initialized: {self.width}x{self.height}")
    
    def pil_to_pygame(self, pil_image):
        """Convert PIL Image to pygame Surface"""
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
        
        raw_image = pil_image.tobytes()
        pygame_image = pygame.image.fromstring(raw_image, pil_image.size, 'RGBA')
        return pygame_image
    
    def display_image(self, image_path_or_pil):
        """Load and display an image"""
        try:
            if isinstance(image_path_or_pil, str):
                # File path
                image = pygame.image.load(image_path_or_pil)
            else:
                # PIL Image
                image = self.pil_to_pygame(image_path_or_pil)
            
            # Scale to fit screen
            orig_width, orig_height = image.get_size()
            scale_x = self.width / orig_width
            scale_y = self.height / orig_height
            scale = min(scale_x, scale_y)
            
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            self.current_image = pygame.transform.scale(image, (new_width, new_height))
            self.image_x = (self.width - new_width) // 2
            self.image_y = (self.height - new_height) // 2
                        
            # Update display immediately
            self.update_display()
            
        except Exception as e:
            print(f"Error loading image: {e}")
    
    def update_display(self):
        """Update the display"""
        self.screen.fill((0, 0, 0))
        if self.current_image:
            self.screen.blit(self.current_image, (self.image_x, self.image_y))
        pygame.display.flip()
    
    def check_for_quit(self):
        """Check for quit events - call this periodically"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return True
        return False
    
    def quit(self):
        """Clean up and quit"""
        pygame.quit()