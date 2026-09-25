//
//  GlassesApp.swift
//  Glasses
//
//  Created by Tristan Varner on 8/18/26.
//

import MWDATCore
import SwiftUI

@main
struct GlassesApp: App {
    init() {
        #if DEBUG
        // Before anything reads `TowerConfiguration`, which fixes the address
        // for the process on first use.
        if ProcessInfo.processInfo.arguments.contains(TowerAddressStore.uiTestResetArgument) {
            TowerAddressStore().clear()
        }
        // Before the root view decides whether to show the first-run cards.
        if ProcessInfo.processInfo.arguments.contains(OnboardingStore.uiTestResetArgument) {
            OnboardingStore().reset()
        }
        #endif
        do {
            try Wearables.configure()
        } catch {
            #if DEBUG
            print("[Glasses] Failed to configure Wearables SDK: \(error)")
            #endif
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                // Meta AI calls back into the app during registration/permission
                // flows via the glasses:// scheme with a metaWearablesAction
                // query item. Anything else is ignored.
                .onOpenURL { url in
                    guard
                        let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
                        components.queryItems?.contains(where: { $0.name == "metaWearablesAction" }) == true
                    else {
                        return
                    }
                    Task {
                        _ = try? await Wearables.shared.handleUrl(url)
                    }
                }
        }
    }
}
